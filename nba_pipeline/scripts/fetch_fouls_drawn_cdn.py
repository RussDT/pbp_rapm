"""Fetch the NBA live-CDN play-by-play foul ledger, including who drew each foul.

The stats.nba.com v3 feed behind ``raw_data/NBA{YY}.parquet`` names only the
fouler (the trailing parenthetical is the referee) and the retired v2 feed that
carried the fouled player now returns empty bodies for every game. The static
CDN feed ``cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json``
carries ``foulDrawnPersonId`` on every non-technical foul from 2019-20 on
(2018-19 and earlier return 403), and reconciles exactly with box-score
personal fouls drawn (``PFD``). This script fetches the games already present in
``player_game_lines.parquet`` for seasons >= 2020, keeps only foul actions, and
writes a resumable ledger plus a per-game status table. A game that could not
be fetched is recorded as unavailable and stays unscored downstream; it is
never treated as zero fouls.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"nba_pipeline/results/estimated_skills"
LINES = RESULTS/"player_game_lines.parquet"
OUT = RESULTS/"fouls_drawn"
CDN = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
FIRST_SEASON = 2020   # 2019-20; the CDN returns 403 for earlier games
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com", "Accept": "*/*"}
LEDGER_COLUMNS = ["game_id", "action_number", "order_number", "period", "clock", "time_actual", "team_id",
                  "fouler_id", "fouler_name", "drawn_id", "drawn_name", "sub_type", "descriptor", "qualifiers",
                  "official_id", "description"]


def game_universe(lines_path=LINES, first_season=FIRST_SEASON):
    lines = pd.read_parquet(lines_path, columns=["game_id", "season", "season_type", "game_date"])
    games = lines[lines.season.ge(first_season)].drop_duplicates("game_id")
    return games.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)


def _int(value):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def foul_rows(game_id, actions):
    """Foul actions only. ``drawn_id`` is None when the feed names no fouled player (technicals)."""
    rows = []
    for a in actions:
        if str(a.get("actionType", "")).lower() != "foul":
            continue
        qualifiers = a.get("qualifiers") or []
        rows.append({
            "game_id": str(game_id), "action_number": _int(a.get("actionNumber")),
            "order_number": _int(a.get("orderNumber")), "period": _int(a.get("period")),
            "clock": a.get("clock"), "time_actual": a.get("timeActual"), "team_id": _int(a.get("teamId")),
            "fouler_id": _int(a.get("personId")), "fouler_name": a.get("playerName"),
            "drawn_id": _int(a.get("foulDrawnPersonId")), "drawn_name": a.get("foulDrawnPlayerName"),
            "sub_type": (a.get("subType") or None), "descriptor": (a.get("descriptor") or None),
            "qualifiers": "|".join(str(q) for q in qualifiers) if qualifiers else None,
            "official_id": _int(a.get("officialId")), "description": a.get("description")})
    return rows


def parse_payload(content):
    """Decode a CDN payload. A handful of 2019-20 objects (e.g. 0021900777, 0021900795) are stored as a
    Python-repr dict with single quotes rather than JSON; ``ast.literal_eval`` reads those exactly."""
    text = content.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import ast
        return ast.literal_eval(text)


def fetch_game(game_id, attempts=4, timeout=30, sleep=2.0):
    """Returns (status, actions, http_status, bytes). status in {ok, unavailable, error}."""
    from curl_cffi import requests as cr
    last = None
    for attempt in range(1, attempts+1):
        try:
            r = cr.get(CDN.format(game_id=game_id), headers=HEADERS, timeout=timeout, impersonate="chrome120")
            if r.status_code == 200:
                actions = parse_payload(r.content)["game"]["actions"]
                return "ok", actions, 200, len(r.content)
            if r.status_code in (403, 404):
                return "unavailable", None, r.status_code, len(r.content)
            last = f"http {r.status_code}"
        except Exception as e:   # network / parse
            last = f"{type(e).__name__}: {str(e)[:80]}"
        time.sleep(sleep*attempt)
    return "error", None, None, last


def load_status(out):
    path = out/"game_status.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"game_id": str})
    return pd.DataFrame(columns=["game_id", "season", "season_type", "game_date", "status", "http_status",
                                 "actions", "fouls", "fouls_with_drawn", "technical_fouls", "fetched_at", "detail"])


def run(out, workers=8, limit=None, retry_errors=True, lines_path=LINES):
    out.mkdir(parents=True, exist_ok=True)
    games = game_universe(lines_path)
    status = load_status(out)
    done = set(status.loc[status.status.eq("ok"), "game_id"])
    if not retry_errors:
        done |= set(status.loc[status.status.ne("ok"), "game_id"])
    todo = games[~games.game_id.isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(games)} games in universe, {len(done)} already fetched, {len(todo)} to fetch", flush=True)
    ledger_path = out/"foul_ledger.parquet"
    ledger = pd.read_parquet(ledger_path) if ledger_path.exists() else pd.DataFrame(columns=LEDGER_COLUMNS)
    new_rows, new_status = [], []
    start = time.time()

    def work(row):
        st, actions, http, detail = fetch_game(row.game_id)
        rows = foul_rows(row.game_id, actions) if actions is not None else []
        rec = {"game_id": row.game_id, "season": int(row.season), "season_type": row.season_type,
               "game_date": pd.Timestamp(row.game_date).strftime("%Y-%m-%d"), "status": st, "http_status": http,
               "actions": len(actions) if actions is not None else None, "fouls": len(rows) if actions is not None else None,
               "fouls_with_drawn": sum(1 for r in rows if r["drawn_id"]) if actions is not None else None,
               "technical_fouls": sum(1 for r in rows if r["sub_type"] == "technical") if actions is not None else None,
               "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "detail": None if st == "ok" else detail}
        return rec, rows

    def checkpoint():
        nonlocal ledger, status, new_rows, new_status
        if not new_status:
            return
        got = pd.DataFrame(new_status)
        status = pd.concat([status[~status.game_id.isin(got.game_id)], got], ignore_index=True)
        status = status.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
        status.to_csv(out/"game_status.csv", index=False)
        if new_rows:
            add = pd.DataFrame(new_rows, columns=LEDGER_COLUMNS)
            ledger = pd.concat([ledger[~ledger.game_id.isin(add.game_id)], add], ignore_index=True)
            ledger = ledger.sort_values(["game_id", "order_number", "action_number"], na_position="last").reset_index(drop=True)
            ledger.to_parquet(ledger_path, index=False)
        new_rows, new_status = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, row) for row in todo.itertuples(index=False)]
        for i, fut in enumerate(as_completed(futures), 1):
            rec, rows = fut.result()
            new_status.append(rec)
            new_rows.extend(rows)
            if i % 250 == 0 or i == len(futures):
                checkpoint()
                counts = status.status.value_counts().to_dict()
                print(f"{i}/{len(futures)} fetched in {time.time()-start:.0f}s; status counts {counts}", flush=True)
    checkpoint()
    summary = {"games_in_universe": int(len(games)), "status_counts": status.status.value_counts().to_dict(),
               "ledger_rows": int(len(ledger)), "seasons": sorted(int(s) for s in games.season.unique()),
               "fouls_with_drawn_share": float(status.fouls_with_drawn.sum()/max(status.fouls.sum(), 1)),
               "sub_type_descriptor_counts": ledger.groupby([ledger.sub_type.fillna("<none>"), ledger.descriptor.fillna("<none>")]).size().to_dict()
               if len(ledger) else {}, "source": CDN, "first_season": FIRST_SEASON}
    summary["sub_type_descriptor_counts"] = {f"{k[0]}/{k[1]}": int(v) for k, v in summary["sub_type_descriptor_counts"].items()}
    (out/"fetch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="fetch at most this many outstanding games")
    parser.add_argument("--no-retry-errors", action="store_true", help="do not refetch games whose last status was not ok")
    args = parser.parse_args()
    run(args.out, workers=args.workers, limit=args.limit, retry_errors=not args.no_retry_errors)
