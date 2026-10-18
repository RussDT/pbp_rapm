"""Restartable private NBA BoxScoreMatchupsV3 archive, ending seasons 2018–26.

Raw responses are saved before any normalization. The official game census also
keeps preseason, play-in and cup-final games; feature consumers choose their
phase explicitly. All-Star games are not ordinary NBA team matchups. Downloads
today establish measurement availability, not historical publication vintage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache/matchup_games_v1"
PHASES = {"001": "preseason", "002": "rs", "004": "ps", "005": "playin", "006": "cup_final"}
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://stats.nba.com", "Referer": "https://stats.nba.com/players/matchups/",
    "x-nba-stats-origin": "stats", "x-nba-stats-token": "true",
    "Accept-Encoding": "gzip, deflate, br", "Connection": "keep-alive",
    "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "empty",
}
HTTP_LOCAL = threading.local()


def http_session():
    # Reuse one TLS connection pool per worker rather than reconnecting for
    # every game. The global request interval still applies to every request.
    if not hasattr(HTTP_LOCAL, "session"):
        HTTP_LOCAL.session = requests.Session()
        HTTP_LOCAL.session.headers.update(HEADERS)
    return HTTP_LOCAL.session


def reset_http_session():
    """Drop a failed transport pool before retrying the public request."""
    session = getattr(HTTP_LOCAL, "session", None)
    if session is not None:
        session.close()
        del HTTP_LOCAL.session


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    temp.replace(path)


def official_team_ids(rows):
    """Resolve venue slots from matchup tokens, including shared neutral text.

    Neutral-site LeagueGameFinder rows can both say ``AWAY @ HOME``. Their
    TEAM_ID is not necessarily the team on the left side of that text.
    """
    identities = rows[["TEAM_ABBREVIATION", "TEAM_ID"]].drop_duplicates()
    if identities.TEAM_ABBREVIATION.duplicated().any():
        raise ValueError("Ambiguous official team abbreviation")
    teams = dict(zip(identities.TEAM_ABBREVIATION, identities.TEAM_ID))
    home, away = set(), set()
    for value in rows.MATCHUP:
        match = re.fullmatch(r"(\S+)\s+(vs\.|@)\s+(\S+)", str(value).strip())
        if match is None:
            raise ValueError("Unrecognized official matchup text: " + str(value))
        left, separator, right = match.groups()
        h, a = (left, right) if separator == "vs." else (right, left)
        if h in teams: home.add(int(teams[h]))
        if a in teams: away.add(int(teams[a]))
    if len(home) > 1 or len(away) > 1 or home & away:
        raise ValueError("Conflicting official home/away matchup identities")
    return next(iter(home), 0), next(iter(away), 0)


def census(out, years):
    """Use completed official team games, cross-check dates against canonical PBP."""
    from nba_api.stats.endpoints import leaguegamefinder
    sources = out / "census_sources"
    sources.mkdir(parents=True, exist_ok=True)
    lines = pd.read_parquet(ROOT / "results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet",
                            columns=["game_id", "game_date", "season", "season_type"])
    lines["game_id"] = lines.game_id.astype(str).str.zfill(10)
    canonical = lines.drop_duplicates(["game_id", "game_date", "season", "season_type"])
    if canonical.game_id.duplicated().any():
        raise ValueError("Canonical game ID has conflicting dates/seasons")
    rows = []
    for year in years:
        source = sources / f"official_games_{year}.csv"
        if not source.exists():
            pilot = ROOT / "cache/matchup_game_probe_v1" / source.name
            if pilot.exists():
                shutil.copyfile(pilot, source)
            else:
                for attempt in range(3):
                    try:
                        endpoint = leaguegamefinder.LeagueGameFinder(
                            season_nullable=f"{year-1}-{year%100:02d}", league_id_nullable="00",
                            player_or_team_abbreviation="T", headers=HEADERS, timeout=20)
                        endpoint.get_data_frames()[0].to_csv(source, index=False)
                        break
                    except requests.RequestException:
                        if attempt == 2:
                            raise
                        time.sleep(2**attempt)
                time.sleep(.5)
            write_json(source.with_suffix(".meta.json"), {"fetched_at": utc(), "endpoint": "LeagueGameFinder",
                "ending_season": year, "sha256": sha_bytes(source.read_bytes()),
                "note": "Unfiltered season type; only completed NBA-team games retained."})
        d = pd.read_csv(source, dtype={"GAME_ID": str})
        d["GAME_ID"] = d.GAME_ID.str.zfill(10)
        for gid, g in d.groupby("GAME_ID", sort=True):
            phase = PHASES.get(gid[:3])
            if phase is None or not g.WL.isin(["W", "L"]).any():
                continue
            dates = pd.to_datetime(g.GAME_DATE).dt.normalize().unique()
            if len(dates) != 1 or int(gid[3:5]) != (year-1) % 100:
                raise ValueError(f"Official census conflict {gid}")
            # International preseason opponents need not have a league00 row.
            home, away = official_team_ids(g)
            rows.append(dict(game_id=gid, game_date=pd.Timestamp(dates[0]), season=year, season_type=phase,
                             home_team_id=home, away_team_id=away,
                             official_team_rows=len(g), census_source="official"))
        print(json.dumps({"census_season": year, "team_rows": len(d)}), flush=True)
    frame = pd.DataFrame(rows)
    frame = frame.merge(canonical.rename(columns={"game_date": "pbp_date", "season": "pbp_season",
                        "season_type": "pbp_phase"}), on="game_id", how="outer", validate="one_to_one")
    frame = frame.loc[frame.season.fillna(frame.pbp_season).isin(years)].copy()
    both = frame.game_date.notna() & frame.pbp_date.notna()
    if (frame.loc[both, "game_date"].ne(frame.loc[both, "pbp_date"]).any()
            or frame.loc[both, "season"].ne(frame.loc[both, "pbp_season"]).any()
            or frame.loc[both, "season_type"].ne(frame.loc[both, "pbp_phase"]).any()):
        raise ValueError("Official/PBP date, season or phase conflict")
    frame["in_canonical_pbp"] = frame.pbp_date.notna()
    frame["game_date"] = frame.game_date.fillna(frame.pbp_date)
    frame["season"] = frame.season.fillna(frame.pbp_season).astype(int)
    frame["season_type"] = frame.season_type.fillna(frame.pbp_phase)
    frame["census_source"] = frame.census_source.fillna("canonical_pbp_only")
    for col in ["home_team_id", "away_team_id", "official_team_rows"]:
        frame[col] = frame[col].fillna(0).astype("int64")
    frame = frame.drop(columns=["pbp_date", "pbp_season", "pbp_phase"]).sort_values(["season", "game_date", "game_id"])
    if frame.game_id.duplicated().any() or frame.season.gt(2026).any():
        raise ValueError("Invalid census identity or protected season")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "game_census.parquet", index=False)
    frame.groupby(["season", "season_type"]).size().rename("games").to_csv(out / "census_counts.csv")
    return frame


def normalize(payload, game):
    """Native outer player is offense; nested player is opposing defender."""
    box = payload.get("boxScoreMatchups")
    if not isinstance(box, dict) or str(box.get("gameId", "")).zfill(10) != game["game_id"]:
        raise ValueError("Missing/mismatched boxScoreMatchups gameId")
    rows = []
    teams = [box.get("homeTeam", {}), box.get("awayTeam", {})]
    if not any(t.get("players", []) for t in teams):
        # Uncovered games have valid top-level identity but zero inner teams.
        return pd.DataFrame(), {"status": "empty", "rows": 0}
    rosters = [set(int(p["personId"]) for p in t.get("players", [])) for t in teams]
    for side, team in enumerate(teams):
        other = teams[1-side]
        if game.get("home_team_id", 0) and side == 0 and int(team.get("teamId", 0)) != game["home_team_id"]:
            raise ValueError("Official home team disagrees")
        if game.get("away_team_id", 0) and side == 1 and int(team.get("teamId", 0)) != game["away_team_id"]:
            raise ValueError("Official away team disagrees")
        for attacker in team.get("players", []):
            for defender in attacker.get("matchups", []):
                did = int(defender["personId"])
                if did in rosters[side]:
                    raise ValueError("Nested defender belongs to same native offensive roster")
                row = dict(game_id=game["game_id"], game_date=game["game_date"], season=game["season"],
                           season_type=game["season_type"], offense_team_id=int(team["teamId"]),
                           defense_team_id=int(other["teamId"]), offense_player_id=int(attacker["personId"]),
                           defense_player_id=did,
                           defender_in_opposing_native_roster=did in rosters[1-side],
                           offense_player_name=f"{attacker.get('firstName', '')} {attacker.get('familyName', '')}".strip(),
                           defense_player_name=f"{defender.get('firstName', '')} {defender.get('familyName', '')}".strip())
                row.update(defender.get("statistics", {}))
                rows.append(row)
    if not rows:
        return pd.DataFrame(), {"status": "empty", "rows": 0}
    d = pd.DataFrame(rows)
    keys = ["game_id", "offense_player_id", "defense_player_id"]
    if d.duplicated(keys).any():
        raise ValueError("Duplicate game/offense/defense pair")
    numeric = [c for c in d if c not in keys + ["game_date", "season_type", "offense_player_name", "defense_player_name", "matchupMinutes", "defender_in_opposing_native_roster"]]
    negative = []
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="raise")
        if d[col].lt(0).any():
            negative.append(col)
    # Retain source defects for audit; never silently turn absent stats into zero.
    violations = []
    for made, attempted in [("matchupFieldGoalsMade", "matchupFieldGoalsAttempted"),
                            ("matchupThreePointersMade", "matchupThreePointersAttempted"),
                            ("matchupFreeThrowsMade", "matchupFreeThrowsAttempted"),
                            ("helpFieldGoalsMade", "helpFieldGoalsAttempted")]:
        if made in d and attempted in d and d[made].gt(d[attempted]).any():
            violations.append(made)
    d["source_counts_valid"] = not (negative or violations)
    return d, dict(status="ok", rows=len(d), negative_fields=negative, makes_over_attempts=violations,
                   defenders_absent_opposing_native_roster=int((~d.defender_in_opposing_native_roster).sum()),
                   tracked_partial_possessions=float(d.partialPossessions.sum()) if "partialPossessions" in d else None,
                   tracked_points=float(d.playerPoints.sum()) if "playerPoints" in d else None,
                   tracked_fga=float(d.matchupFieldGoalsAttempted.sum()) if "matchupFieldGoalsAttempted" in d else None,
                   tracked_fta=float(d.matchupFreeThrowsAttempted.sum()) if "matchupFreeThrowsAttempted" in d else None)


class RateLimit:
    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_at = 0.
        self.aborted = threading.Event()

    def wait(self):
        with self.lock:
            if self.aborted.is_set():
                raise PermissionError("Batch stopped after NBA service refusal")
            delay = self.next_at-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if self.aborted.is_set():
                raise PermissionError("Batch stopped after NBA service refusal")
            self.next_at = time.monotonic()+self.interval


def fetch_one(game, out, limiter, retries):
    gid = game["game_id"]
    raw = out / "raw" / f"{gid}.json"
    meta = out / "metadata" / f"{gid}.json"
    dest = out / "pairs" / f"{gid}.parquet"
    previous = json.loads(meta.read_text()) if meta.exists() else {}
    if raw.exists() and previous.get("status") in ["ok", "empty"] and (dest.exists() or previous["status"] == "empty"):
        if previous.get("sha256") != sha_bytes(raw.read_bytes()):
            raise ValueError(f"Cached source hash changed {gid}")
        return {**previous, "cached": True}
    pilot = ROOT / "cache/matchup_game_probe_v1" / raw.name
    if not raw.exists() and pilot.exists():
        shutil.copyfile(pilot, raw)
    error = None
    for attempt in range(retries+1):
        try:
            if limiter.aborted.is_set():
                raise PermissionError("Batch stopped after NBA service refusal")
            if not raw.exists():
                limiter.wait()
                response = http_session().get("https://stats.nba.com/stats/boxscorematchupsv3", params={"GameID": gid},
                                             timeout=(10, 30))
                if response.status_code in [403, 429]:
                    # Stop the batch on service refusal. Restart later rather than bypass.
                    limiter.aborted.set()
                    raise PermissionError(f"NBA service HTTP {response.status_code}; batch must cool down")
                response.raise_for_status()
                payload = response.json()
                # Refuse an HTML/error response before persisting it as a good game.
                if "boxScoreMatchups" not in payload:
                    raise ValueError("Endpoint returned no boxScoreMatchups")
                temp = raw.with_suffix(".json.tmp")
                temp.write_bytes(response.content)
                temp.replace(raw)
            content = raw.read_bytes()
            pairs, result = normalize(json.loads(content), game)
            if len(pairs):
                pairs.to_parquet(dest, index=False)
            result.update(game_id=gid, game_date=str(pd.Timestamp(game["game_date"]).date()),
                          season=game["season"], season_type=game["season_type"], fetched_at=utc(),
                          sha256=sha_bytes(content), raw_bytes=len(content), attempts=attempt+1,
                          endpoint="BoxScoreMatchupsV3", orientation="outer_offense_nested_defense")
            write_json(meta, result)
            return result
        except PermissionError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, requests.RequestException):
                # A fresh ordinary request recovered a game after repeated
                # persistent-session timeouts. Preserve rate/permission gates.
                reset_http_session()
            # A permanently bad cached payload needs review, not endless refetch.
            if raw.exists():
                break
            if attempt < retries:
                time.sleep(min(20, 2**attempt))
    result = dict(game_id=gid, game_date=str(pd.Timestamp(game["game_date"]).date()), season=game["season"],
                  season_type=game["season_type"], status="error", error=error, checked_at=utc())
    write_json(meta, result)
    return result


def audit(out):
    games = pd.read_parquet(out / "game_census.parquet")
    receipts = [json.loads(p.read_text()) for p in sorted((out / "metadata").glob("*.json"))]
    r = pd.DataFrame(receipts)
    if len(r):
        r = games.merge(r.drop(columns=["game_date", "season", "season_type"], errors="ignore"), on="game_id", how="left", validate="one_to_one")
    else:
        r = games.assign(status="pending")
    r["status"] = r.status.fillna("pending")
    r.to_parquet(out / "download_audit.parquet", index=False)
    counts = r.groupby(["season", "season_type", "status"]).size().rename("games").reset_index()
    counts.to_csv(out / "download_counts.csv", index=False)
    summary = dict(updated_at=utc(), census_games=len(games), status=r.status.value_counts().to_dict(),
                   pair_rows=int(pd.to_numeric(r.get("rows", pd.Series(dtype=float)), errors="coerce").sum()),
                   census_sha256=sha_bytes((out / "game_census.parquet").read_bytes()),
                   protected_season=2027, source_vintage="retrospective retrieval; original publication vintages uncertified")
    write_json(out / "progress.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def download(out, frame, workers, interval, retries, limit):
    for name in ["raw", "pairs", "metadata"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    records = frame.to_dict("records")
    # Avoid scheduling thousands of already-cached games on resume.
    pending = []
    for g in records:
        meta = out / "metadata" / f"{g['game_id']}.json"
        done = json.loads(meta.read_text()) if meta.exists() else {}
        if done.get("status") not in ["ok", "empty"]:
            pending.append(g)
    if limit:
        pending = pending[:limit]
    limiter = RateLimit(interval)
    start = time.monotonic()
    print(json.dumps({"download_pending": len(pending), "workers": workers, "interval": interval}), flush=True)
    # At most one active game per worker. A slow retry must not hold up an
    # otherwise completed batch; refusals stop replenishing the bounded queue.
    iterator = iter(pending)
    completed = 0
    recent = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = set()

        def replenish():
            game = next(iterator, None)
            if game is not None:
                futures.add(pool.submit(fetch_one, game, out, limiter, retries))

        for _ in range(workers):
            replenish()
        while futures:
            done, _ = wait_futures(futures, return_when=FIRST_COMPLETED)
            try:
                for future in done:
                    futures.remove(future)
                    result = future.result()
                    completed += 1
                    recent.append(result['status'])
                    if completed == 1 or completed % 200 == 0 or completed == len(pending):
                        print(json.dumps({"processed_this_run": completed,
                            "elapsed_seconds": round(time.monotonic()-start, 1),
                            "recent_status": pd.Series(recent).value_counts().to_dict()}), flush=True)
                        audit(out)
                        recent.clear()
                    replenish()
            except PermissionError:
                limiter.aborted.set()
                for f in futures:
                    f.cancel()
                audit(out)
                raise
    return audit(out)


def renormalize(out):
    """Reparse cached responses after an audited schema/validation correction."""
    frame = pd.read_parquet(out / "game_census.parquet")
    for game in frame.to_dict("records"):
        gid = game["game_id"]
        raw = out / "raw" / f"{gid}.json"
        if not raw.exists():
            continue
        meta = out / "metadata" / f"{gid}.json"
        old = json.loads(meta.read_text()) if meta.exists() else {}
        content = raw.read_bytes()
        if old.get("sha256") and old["sha256"] != sha_bytes(content):
            raise ValueError("Cached raw source hash changed: " + gid)
        try:
            pairs, result = normalize(json.loads(content), game)
            if len(pairs):
                pairs.to_parquet(out / "pairs" / f"{gid}.parquet", index=False)
            result.update(game_id=gid, game_date=str(pd.Timestamp(game["game_date"]).date()),
                          season=game["season"], season_type=game["season_type"],
                          fetched_at=old.get("fetched_at", None), normalized_at=utc(),
                          response_generated_at=json.loads(content).get("meta", {}).get("time"),
                          sha256=sha_bytes(content), raw_bytes=len(content), endpoint="BoxScoreMatchupsV3",
                          orientation="outer_offense_nested_defense")
            write_json(meta, result)
        except ValueError as exc:
            old.update(status="error", error=str(exc), checked_at=utc(), sha256=sha_bytes(content))
            write_json(meta, old)
    audit(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["census", "fetch", "normalize", "audit", "all"], default="all")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2018, 2027)))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not args.years or min(args.years) < 2018 or max(args.years) > 2026:
        parser.error("Only ending seasons2018–2026;2027 is protected")
    if args.workers not in [1, 2, 3, 4] or args.interval < .1:
        parser.error("Require1–4workers and >=.1seconds between requests")
    if args.stage in ["census", "all"]:
        frame = census(args.out, args.years)
    elif args.stage == "fetch":
        frame = pd.read_parquet(args.out / "game_census.parquet")
        frame = frame.loc[frame.season.isin(args.years)]
    if args.stage in ["fetch", "all"]:
        download(args.out, frame, args.workers, args.interval, args.retries, args.limit)
    elif args.stage == "audit":
        audit(args.out)
    elif args.stage == "normalize":
        renormalize(args.out)


if __name__ == "__main__":
    main()
