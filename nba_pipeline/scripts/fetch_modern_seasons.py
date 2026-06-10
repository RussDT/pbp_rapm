#!/usr/bin/env python3
"""
Careful, resumable driver for re-fetching modern NBA seasons (PBP + rotations).

Wraps 01_fetch_pbp_data.py in small UPDATE-mode chunks so that:
  - progress is checkpointed to NBA{YY}{_PS}.parquet after every chunk,
  - an interruption only loses the in-flight chunk (already-fetched games are
    skipped on the next run via update mode),
  - request volume stays gentle to avoid the stats.nba.com (Akamai) soft-block.

Each chunk = one invocation of 01_fetch_pbp_data.py in update mode with --limit,
which fetches up to CHUNK *missing* games for that season, appends, and exits.

Default is DIRECT (no proxy), since stats.nba.com works fine from a residential
IP as long as you don't burst it.

Usage:
    # Refill everything 2014-2025 (NBA14..NBA25), RS + PS, direct, gentle:
    python fetch_modern_seasons.py --start 14 --end 25 --season-types RS PS

    # Single season smoke test:
    python fetch_modern_seasons.py --start 25 --end 25 --season-types RS --max-chunks 1

    # Use the proxy instead of direct:
    python fetch_modern_seasons.py --start 14 --end 25 --proxy

Year mapping: 14 -> 2013-14 ... 25 -> 2024-25 ... 26 -> 2025-26.
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from curl_cffi import requests as creq  # Chrome TLS fingerprint; beats Akamai
    _HAS_CURL_CFFI = True
except Exception:  # pragma: no cover
    import requests as creq
    _HAS_CURL_CFFI = False

SCRIPT_DIR = Path(__file__).parent
FETCH_SCRIPT = SCRIPT_DIR / "01_fetch_pbp_data.py"
RAW_DIR = SCRIPT_DIR.parent / "raw_data"
LOG_DIR = SCRIPT_DIR.parent / "logs"

# Phrases the inner script prints when a season has nothing left to fetch.
DONE_MARKERS = ("Already up to date", "No new games to fetch", "No games found")

PROBE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
PROBE_URL = ("https://stats.nba.com/stats/leaguegamelog?Counter=0&Direction=DESC"
             "&LeagueID=00&PlayerOrTeam=T&Season=2024-25&SeasonType=Regular+Season&Sorter=DATE")


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def stats_reachable(timeout: int = 25) -> bool:
    try:
        if _HAS_CURL_CFFI:
            r = creq.get(PROBE_URL, headers=PROBE_HEADERS, timeout=timeout,
                         impersonate="chrome120")
        else:
            r = creq.get(PROBE_URL, headers=PROBE_HEADERS, timeout=timeout)
        return r.status_code == 200 and len(r.text) > 1000
    except Exception:
        return False


def wait_until_reachable(max_wait_min: int) -> bool:
    """Probe stats.nba.com, backing off until it answers or we give up."""
    if stats_reachable():
        log("stats.nba.com reachable - starting.")
        return True
    log("stats.nba.com not reachable (likely soft-blocked). Waiting for it to clear...")
    waited = 0.0
    delay = 60.0
    while waited < max_wait_min * 60:
        log(f"  sleeping {int(delay)}s before re-probing (waited {int(waited/60)}m so far)...")
        time.sleep(delay)
        waited += delay
        if stats_reachable():
            log(f"stats.nba.com reachable after ~{int(waited/60)}m - starting.")
            return True
        delay = min(delay * 1.5, 600.0)  # back off up to 10 min
    log(f"Gave up waiting after {max_wait_min}m. stats.nba.com still blocked.")
    return False


def fetch_season(yy: str, stype: str, args, logfile: Path) -> bool:
    """Run chunked update-mode fetches for one season/type until done."""
    ps_suffix = "_PS" if stype == "PS" else ""
    out_path = RAW_DIR / f"NBA{yy}{ps_suffix}.parquet"
    log(f"=== Season {yy} {stype} -> {out_path.name} ===")

    prev_count = count_games(out_path)
    stalls = 0  # consecutive chunks that added zero new games
    season_total = 0  # max "of N total" seen from the game-list (true season size)

    for chunk in range(1, args.max_chunks + 1):
        cmd = [
            sys.executable, "-u", str(FETCH_SCRIPT), yy, stype,
            "--limit", str(args.chunk),
            "--workers", str(args.workers),
        ]
        if not args.proxy:
            cmd.append("--no-proxy")
        log(f"  chunk {chunk}/{args.max_chunks}: {' '.join(cmd[2:])}")

        with open(logfile, "a") as lf:
            lf.write(f"\n\n########## {yy} {stype} chunk {chunk} "
                     f"{datetime.now().isoformat()} ##########\n")
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)

        # Read back the tail of the log to detect completion / progress.
        tail = ""
        try:
            tail = out_path_tail(logfile)
        except Exception:
            pass

        if proc.returncode != 0:
            log(f"  chunk returned non-zero ({proc.returncode}); see {logfile.name}. "
                f"Cooling down {args.error_sleep}s.")
            time.sleep(args.error_sleep)
            if not stats_reachable():
                if not wait_until_reachable(args.max_wait_min):
                    return False
            continue

        # Track the true season size from the game-list ("of N total").
        for m in re.findall(r"of (\d+) total", tail):
            season_total = max(season_total, int(m))

        n = count_games(out_path)

        if any(marker in tail for marker in DONE_MARKERS):
            # Guard against a truncated/flaky leaguegamelog returning a short
            # list (all-present -> false "no new games"). Only trust completion
            # if we actually have ~all games the season is known to contain.
            if season_total and n < season_total - 2:
                log(f"  '{out_path.name}' reported done but only {n}/{season_total} "
                    f"games present; likely a truncated game-list. Treating as a "
                    f"stall and retrying.")
                stalls += 1
                if stalls >= args.max_stalls:
                    log(f"  Season {yy} {stype} stuck at {n}/{season_total} after "
                        f"{stalls} bad passes; moving on. Re-run later to fill the rest.")
                    return True
                time.sleep(min(args.stall_sleep * stalls, 600))
                continue
            log(f"  Season {yy} {stype} COMPLETE ({n}/{season_total or n} games).")
            return True

        gained = n - prev_count
        prev_count = n

        if gained > 0:
            stalls = 0
            log(f"  checkpoint: {out_path.name} now has {n} games (+{gained}).")
            time.sleep(args.chunk_sleep)
        else:
            stalls += 1
            # gamerotation 500s are throttle/cache-cold; a longer cooldown lets
            # the endpoint recover so the same games succeed on the next pass.
            cooldown = min(args.stall_sleep * stalls, 600)
            log(f"  no new games this chunk (stall {stalls}/{args.max_stalls}); "
                f"cooling down {cooldown}s to let gamerotation recover.")
            if stalls >= args.max_stalls:
                log(f"  Season {yy} {stype} stalled at {n} games after "
                    f"{stalls} empty passes; moving on. Re-run later to fill the rest.")
                return True
            time.sleep(cooldown)

    log(f"  Reached max-chunks ({args.max_chunks}) for {yy} {stype}; "
        f"{count_games(out_path)} games. Re-run to continue.")
    return True


def out_path_tail(logfile: Path, nbytes: int = 4000) -> str:
    with open(logfile, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "replace")


def count_games(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_parquet(out_path, columns=["game_id"])
        return df["game_id"].astype(str).nunique()
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, required=True, help="Start 2-digit year (e.g. 14)")
    ap.add_argument("--end", type=int, required=True, help="End 2-digit year inclusive (e.g. 25)")
    ap.add_argument("--season-types", nargs="+", default=["RS", "PS"], choices=["RS", "PS"])
    ap.add_argument("--chunk", type=int, default=40, help="Games per chunk (default 40)")
    ap.add_argument("--workers", type=int, default=3, help="Parallel workers (default 3)")
    ap.add_argument("--chunk-sleep", type=int, default=20, help="Seconds between chunks (default 20)")
    ap.add_argument("--season-sleep", type=int, default=60, help="Seconds between seasons (default 60)")
    ap.add_argument("--error-sleep", type=int, default=120, help="Seconds to wait after a failed chunk")
    ap.add_argument("--max-chunks", type=int, default=120, help="Safety cap on chunks per season")
    ap.add_argument("--max-stalls", type=int, default=6,
                    help="Consecutive zero-progress chunks before moving on")
    ap.add_argument("--stall-sleep", type=int, default=90,
                    help="Base cooldown (s) on a zero-progress chunk; escalates per stall")
    ap.add_argument("--max-wait-min", type=int, default=180,
                    help="Max minutes to wait for stats.nba.com to unblock")
    ap.add_argument("--proxy", action="store_true", help="Use the built-in proxy instead of direct")
    ap.add_argument("--newest-first", action="store_true",
                    help="Fetch most recent seasons first (e.g. 25 before 14)")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"fetch_modern_{args.start}_{args.end}_{stamp}.log"

    years = [f"{y:02d}" for y in range(args.start, args.end + 1)]
    if args.newest_first:
        years = years[::-1]
    log(f"Driver start. Years={years} types={args.season_types} "
        f"chunk={args.chunk} workers={args.workers} proxy={args.proxy}")
    log(f"Inner-script output -> {logfile}")

    if not wait_until_reachable(args.max_wait_min):
        sys.exit(2)

    todo = [(yy, st) for yy in years for st in args.season_types]
    for i, (yy, st) in enumerate(todo, 1):
        log(f"--- [{i}/{len(todo)}] season {yy} {st} ---")
        ok = fetch_season(yy, st, args, logfile)
        if not ok:
            log("Aborting: stats.nba.com unreachable beyond max wait.")
            sys.exit(2)
        if i < len(todo):
            time.sleep(args.season_sleep)

    log("ALL DONE. Final raw_data game counts:")
    for yy in years:
        for st in args.season_types:
            ps = "_PS" if st == "PS" else ""
            p = RAW_DIR / f"NBA{yy}{ps}.parquet"
            log(f"  {p.name}: {count_games(p)} games")


if __name__ == "__main__":
    main()
