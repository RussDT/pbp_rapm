"""Phase 0 frozen pre-tip WNBA forecast emitter.

Contract:
- Schedule source is the keyless public ESPN scoreboard
  (``common.ESPN_SCOREBOARD_URL``); by default only events with ESPN state
  ``'pre'`` get one ``pretip_forecast`` record — in-progress (``'in'``) events
  are skipped with a printed line unless ``--include-live`` is passed. The
  ESPN event state is stored honestly on every record as ``espn_state`` so
  scoring can enforce the pre-tip fence. Team mapping goes through
  ``common.espn_to_cdn_tricode`` and FAILS CLOSED on unknown teams.
- Ratings come from the live board (``common.load_current_board``, positive
  good on both sides, ``net_rapm = off + def`` per 100); minutes come from
  ``project_wnba_minutes.project_team_minutes`` on the game-log panel at
  ``common.GAME_LOG_PATH`` with ``as_of_date`` = the game date. If a team has
  no projectable history the fallback is a uniform 200-minute split
  (``minutes_source = "uniform_fallback"``) over a roster derived from each
  player's most recent game-log appearance, then — only when the franchise
  has no game-log presence at all — the ``common.load_team_assignments``
  season roster.
- Margin algebra (pinned): ``team_per100 = sum(net_rapm_i * proj_minutes_i)
  / 40`` over the 200-minute allocation, players missing from the board
  contribute 0 and are excluded from the rated-minutes share;
  ``margin_mean = (home_per100 - away_per100) * pace / 100 + hca``;
  ``p_home_win = Phi(margin_mean / sigma)``.
- Params come from ``Results/gameday/forecast_params.json`` when present
  (pace = latest fitted season strictly before the game season), else the
  frozen defaults ``hca=2.0 sigma=13.5 pace=80.0`` with ``default_params``
  set true.
- Staleness gate: refuse (nonzero exit) when the game-log panel's max
  ``game_date`` is more than ``STALE_MAX_DAYS`` before the forecast date,
  unless ``--allow-stale``.
- Records append ONLY through ``common.append_log_record`` (append-only,
  hash-chained JSONL); after appending, ``common.verify_log_chain`` must pass
  or the run fails loudly. Re-emitting a game later is allowed — scoring
  selects the last record per game with ``espn_state == 'pre'`` whose
  ``emitted_at_utc`` is strictly before tipoff, so a post-tip re-emission can
  never supersede the true pre-tip forecast.
- Standing fences: market-blind (no odds inputs anywhere in this script);
  2026 is scored with frozen parameters only; no proprietary ratings; no
  credentials; generated JSONL artifacts stay out of git.

Usage:
    .venv/bin/python wnba_test/emit_wnba_pretip_forecasts.py --date 2026-08-10 --dry-run
    .venv/bin/python wnba_test/emit_wnba_pretip_forecasts.py --date 2026-08-10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import pandas as pd

import wnba_gameday_common as common
from project_wnba_minutes import FRANCHISE_ALIASES, TEAM_MINUTES, project_team_minutes

try:  # pragma: no cover - import guard mirrors fetch_wnba_cdn
    from curl_cffi import requests as cffi_requests
except ImportError:  # pragma: no cover
    cffi_requests = None

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

MODEL_VERSION = "phase0_v1"
DEFAULT_HCA_PTS = 2.0
DEFAULT_SIGMA_MARGIN = 13.5
DEFAULT_PACE = 80.0
LOW_COVERAGE_SHARE = 0.85
STALE_MAX_DAYS = 2

PARAMS_PATH = common.GAMEDAY_DIR / "forecast_params.json"
DEFAULT_LOG_PATH = common.FORECAST_LOG_DIR / "forecasts_2026.jsonl"

FetchJson = Callable[[str], "dict[str, Any]"]


# --------------------------------------------------------------------------
# Fetching (network lives only here; everything below is deterministic)
# --------------------------------------------------------------------------


def fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    """GET a public keyless ESPN endpoint and return parsed JSON (raises on failure)."""
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/json,text/plain,*/*"}
    if cffi_requests is not None:
        response = cffi_requests.get(
            url, timeout=timeout, headers=headers, impersonate="chrome"
        )
        if response.status_code != 200:
            raise RuntimeError(f"failed to fetch {url}: HTTP {response.status_code}")
        return response.json()
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:  # nosec - keyless public API
        return json.loads(response.read().decode("utf-8"))


# --------------------------------------------------------------------------
# Params / inputs
# --------------------------------------------------------------------------


def load_forecast_params(game_season: int, params_path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load frozen forecast params; fall back to spec defaults when absent.

    Pace is the latest fitted season strictly before ``game_season`` (2026
    games use 2025 pace) so the confirmation season never informs itself.
    ``default_params`` is true whenever any of hca/sigma/pace fell back.
    """
    hca, sigma, pace = DEFAULT_HCA_PTS, DEFAULT_SIGMA_MARGIN, DEFAULT_PACE
    default_params = True
    if params_path.exists():
        payload = json.loads(params_path.read_text())
        hca = float(payload.get("hca_pts", DEFAULT_HCA_PTS))
        sigma = float(payload.get("sigma_margin", DEFAULT_SIGMA_MARGIN))
        pace_by_season = payload.get("pace_by_season") or {}
        prior_seasons = [int(s) for s in pace_by_season if int(s) < game_season]
        if prior_seasons:
            pace = float(pace_by_season[str(max(prior_seasons))])
            default_params = False
    return {"hca_pts": hca, "sigma_margin": sigma, "pace": pace, "default_params": default_params}


def load_game_log_panel() -> pd.DataFrame:
    logs = pd.read_csv(common.GAME_LOG_PATH, dtype={"game_id": str})
    logs["person_id"] = logs["person_id"].astype(int)
    logs["minutes"] = logs["minutes"].astype(float)
    return logs


def check_panel_staleness(logs: pd.DataFrame, forecast_date: str, allow_stale: bool) -> None:
    max_game_date = str(logs["game_date"].max())
    gap_days = (date.fromisoformat(forecast_date) - date.fromisoformat(max_game_date)).days
    if gap_days > STALE_MAX_DAYS and not allow_stale:
        raise SystemExit(
            f"REFUSED: game-log panel is stale (max game_date {max_game_date} is "
            f"{gap_days} days before forecast date {forecast_date}, limit "
            f"{STALE_MAX_DAYS}). Rebuild the panel with build_wnba_game_logs.py "
            "or pass --allow-stale."
        )


def load_cdn_game_lookup() -> dict[tuple, str]:
    """(game_date, home_tricode, away_tricode) -> CDN game id; ambiguous keys dropped."""
    if not common.GAMES_TABLE_PATH.exists():
        return {}
    games = pd.read_csv(common.GAMES_TABLE_PATH, dtype={"game_id": str})
    lookup: dict[tuple, str] = {}
    ambiguous: set = set()
    for row in games.itertuples(index=False):
        key = (str(row.game_date), str(row.home_tricode), str(row.away_tricode))
        if key in lookup and lookup[key] != row.game_id:
            ambiguous.add(key)
        lookup[key] = str(row.game_id)
    for key in ambiguous:
        lookup.pop(key, None)
    return lookup


# --------------------------------------------------------------------------
# Scoreboard parsing (pure)
# --------------------------------------------------------------------------


def parse_nonfinal_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the scoreboard into per-event schedule dicts, skipping finals.

    Unknown teams raise (fail closed). Returns dicts with espn_event_id,
    home/away CDN tricodes, tipoff_utc, and state ('pre' or 'in').
    """
    events: list[dict[str, Any]] = []
    for event in payload.get("events") or []:
        competitions = event.get("competitions") or []
        if not competitions:
            raise ValueError(f"event {event.get('id')!r}: no competitions block")
        competition = competitions[0]
        home = away = None
        for side in competition.get("competitors") or []:
            if side.get("homeAway") == "home":
                home = side
            elif side.get("homeAway") == "away":
                away = side
        if home is None or away is None:
            raise ValueError(f"event {event.get('id')!r}: missing home/away competitor")
        status_type = (event.get("status") or {}).get("type") or {}
        state = status_type.get("state")
        if state == "post" or status_type.get("completed"):
            continue
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        events.append(
            {
                "espn_event_id": str(event.get("id")),
                "home_tricode": common.espn_to_cdn_tricode(
                    home_team.get("abbreviation"), home_team.get("displayName")
                ),
                "away_tricode": common.espn_to_cdn_tricode(
                    away_team.get("abbreviation"), away_team.get("displayName")
                ),
                "tipoff_utc": str(event.get("date") or competition.get("date") or ""),
                "state": state,
            }
        )
    events.sort(key=lambda row: (row["tipoff_utc"], row["espn_event_id"]))
    return events


# --------------------------------------------------------------------------
# Forecast math (pure)
# --------------------------------------------------------------------------


def latest_team_by_player(logs: pd.DataFrame, game_date: str) -> pd.Series:
    """person_id -> team_tricode of each player's most recent game-log row
    strictly before ``game_date`` (roster rows count: a DNP/INACTIVE row still
    evidences game-day roster membership)."""
    hist = logs.loc[logs["game_date"] < game_date]
    if hist.empty:
        return pd.Series(dtype=object)
    hist = hist.sort_values(["game_date", "game_id"], kind="mergesort")
    return hist.groupby("person_id")["team_tricode"].last()


def team_minutes_frame(
    logs: pd.DataFrame,
    assignments: pd.DataFrame,
    tricode: str,
    game_date: str,
) -> tuple[pd.DataFrame, str]:
    """200-minute allocation for one team; returns (frame, minutes_source).

    Fallback order when the minutes projection refuses: (1) uniform split over
    players whose most recent game-log appearance is with this franchise —
    immune to stale trade rows; (2) only when the franchise has no game-log
    presence at all, the season team-assignment table, which can list traded
    players under a stale first team (see ``common.load_team_assignments``).
    """
    try:
        frame = project_team_minutes(logs, tricode, as_of_date=game_date)
        return frame[["person_id", "proj_minutes"]], "projection"
    except ValueError:
        codes = FRANCHISE_ALIASES.get(tricode, (tricode,))
        latest_team = latest_team_by_player(logs, game_date)
        roster = sorted(
            int(pid) for pid, team in latest_team.items() if team in codes
        )
        if not roster:
            # load_team_assignments frames carry player_id; tolerate person_id
            # so synthetic panels can be passed straight through.
            id_col = "player_id" if "player_id" in assignments.columns else "person_id"
            roster = sorted(
                int(pid)
                for pid in assignments.loc[
                    assignments["team_tricode"].isin(codes), id_col
                ]
            )
        if not roster:
            raise SystemExit(
                f"REFUSED: no minutes projection, no game-log roster, and no "
                f"team-assignment roster for {tricode} as of {game_date}; "
                f"cannot allocate 200 minutes"
            )
        frame = pd.DataFrame(
            {"person_id": roster, "proj_minutes": TEAM_MINUTES / len(roster)}
        )
        return frame, "uniform_fallback"


def team_per100_and_share(
    minutes: pd.DataFrame, net_by_pid: pd.Series
) -> tuple[float, float]:
    """Pinned algebra: per100 = sum(net_rapm_i * proj_minutes_i) / 40.

    Players missing from the board contribute 0 and are excluded from the
    rated-minutes share.
    """
    rated = minutes["person_id"].map(net_by_pid)
    per100 = float((rated.fillna(0.0) * minutes["proj_minutes"]).sum() / (TEAM_MINUTES / 5.0))
    rated_minutes = float(minutes.loc[rated.notna(), "proj_minutes"].sum())
    return per100, rated_minutes / TEAM_MINUTES


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def build_forecast_record(
    event: dict[str, Any],
    game_date: str,
    logs: pd.DataFrame,
    assignments: pd.DataFrame,
    net_by_pid: pd.Series,
    params: dict[str, Any],
    cdn_lookup: dict[tuple, str],
    board_path: Path,
    board_mtime_utc: str,
) -> dict[str, Any]:
    home_minutes, home_source = team_minutes_frame(
        logs, assignments, event["home_tricode"], game_date
    )
    away_minutes, away_source = team_minutes_frame(
        logs, assignments, event["away_tricode"], game_date
    )
    minutes_source = (
        "projection"
        if home_source == "projection" and away_source == "projection"
        else "uniform_fallback"
    )
    home_per100, home_share = team_per100_and_share(home_minutes, net_by_pid)
    away_per100, away_share = team_per100_and_share(away_minutes, net_by_pid)
    margin_mean = (home_per100 - away_per100) * params["pace"] / 100.0 + params["hca_pts"]
    p_home_win = normal_cdf(margin_mean / params["sigma_margin"])
    key = (game_date, event["home_tricode"], event["away_tricode"])
    # Forward-compatible schema: scoring ignores unknown fields, so new keys
    # may be added later without invalidating old hash-chained records.
    # Reserved (documented here, never populated by this emitter) execution-
    # telemetry fields for the future betting layer: market_type,
    # requested_stake, accepted_stake, limit_event.
    return {
        "type": "pretip_forecast",
        "emitted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game_date": game_date,
        "espn_event_id": event["espn_event_id"],
        "cdn_game_id": cdn_lookup.get(key),
        "home_tricode": event["home_tricode"],
        "away_tricode": event["away_tricode"],
        "tipoff_utc": event["tipoff_utc"],
        "espn_state": event.get("state"),
        "board_path": str(board_path),
        "board_mtime_utc": board_mtime_utc,
        "minutes_source": minutes_source,
        "home_per100": round(home_per100, 4),
        "away_per100": round(away_per100, 4),
        "pace_possessions": round(float(params["pace"]), 4),
        "hca_pts": round(float(params["hca_pts"]), 4),
        "margin_mean": round(margin_mean, 4),
        "margin_sigma": round(float(params["sigma_margin"]), 4),
        "p_home_win": round(p_home_win, 6),
        "home_rated_minutes_share": round(home_share, 4),
        "away_rated_minutes_share": round(away_share, 4),
        "low_coverage": bool(
            home_share < LOW_COVERAGE_SHARE or away_share < LOW_COVERAGE_SHARE
        ),
        "default_params": bool(params["default_params"]),
        "model_version": MODEL_VERSION,
    }


# --------------------------------------------------------------------------
# Emission flow
# --------------------------------------------------------------------------


def emit_forecasts(
    game_date: str,
    log_path: Path,
    dry_run: bool,
    allow_stale: bool,
    fetch: FetchJson = fetch_json,
    include_live: bool = False,
) -> list[dict[str, Any]]:
    logs = load_game_log_panel()
    check_panel_staleness(logs, game_date, allow_stale)

    payload = fetch(common.ESPN_SCOREBOARD_URL.format(yyyymmdd=game_date.replace("-", "")))
    events = parse_nonfinal_events(payload)
    if not include_live:
        kept = []
        for event in events:
            if event["state"] == "pre":
                kept.append(event)
            else:
                print(
                    f"{game_date} {event['away_tricode']} @ {event['home_tricode']} "
                    f"(espn {event['espn_event_id']}): SKIPPED — state "
                    f"{event['state']!r} is not 'pre' (pass --include-live to "
                    f"emit anyway; the state is always recorded honestly)"
                )
        events = kept
    if not events:
        print(f"{game_date}: no emittable ESPN events; nothing to emit")
        return []

    board = common.load_current_board()
    net_by_pid = board.set_index("player_id")["net_rapm"].astype(float)
    game_season = int(game_date[:4])
    assignments = common.load_team_assignments(game_season)
    params = load_forecast_params(game_season)
    cdn_lookup = load_cdn_game_lookup()
    board_mtime_utc = datetime.fromtimestamp(
        common.CURRENT_BOARD_PATH.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds")

    records = []
    for event in events:
        record = build_forecast_record(
            event,
            game_date,
            logs,
            assignments,
            net_by_pid,
            params,
            cdn_lookup,
            common.CURRENT_BOARD_PATH,
            board_mtime_utc,
        )
        records.append(record)
        print(
            f"{game_date} {record['away_tricode']} @ {record['home_tricode']} "
            f"(espn {record['espn_event_id']}, state {event['state']}): "
            f"margin {record['margin_mean']:+.2f}, p_home {record['p_home_win']:.3f}, "
            f"coverage {record['home_rated_minutes_share']:.3f}/"
            f"{record['away_rated_minutes_share']:.3f}, "
            f"minutes {record['minutes_source']}"
            f"{' [LOW COVERAGE]' if record['low_coverage'] else ''}"
            f"{' [DEFAULT PARAMS]' if record['default_params'] else ''}"
        )

    if dry_run:
        print(f"dry run: {len(records)} forecast(s) NOT appended to {log_path}")
        return records

    for record in records:
        common.append_log_record(log_path, record)
    ok, n_records, reason = common.verify_log_chain(log_path)
    if not ok:
        raise SystemExit(
            f"FATAL: forecast log chain verification FAILED after append "
            f"({log_path}): {reason}"
        )
    print(f"appended {len(records)} forecast(s) to {log_path}; chain ok ({n_records} records)")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="forecast date YYYY-MM-DD (default: today local)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print forecasts without appending"
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"hash-chained forecast JSONL (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="bypass the game-log panel staleness gate (use knowingly)",
    )
    parser.add_argument(
        "--include-live",
        action="store_true",
        help=(
            "also emit for in-progress ('in') events; the ESPN state is "
            "recorded honestly and scoring will exclude non-pre records"
        ),
    )
    args = parser.parse_args()
    try:
        date.fromisoformat(args.date)
    except ValueError:
        parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    emit_forecasts(
        args.date, args.log_path, args.dry_run, args.allow_stale,
        include_live=args.include_live,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
