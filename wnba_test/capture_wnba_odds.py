"""Capture ESPN WNBA odds snapshots into the hash-chained odds log.

Contract:
- Source is the keyless public ESPN scoreboard (``common.ESPN_SCOREBOARD_URL``).
  When an event embeds no odds (typical once a game is final), fall back to the
  event summary's ``pickcenter[0]`` (``common.ESPN_SUMMARY_URL``).
- ``spread_home`` is the HOME line: negative means the home team is favored,
  so ``market_home_margin = -spread_home``. Verified empirically (August 2026):
  the sign of ESPN's top-level ``spread`` agreed with the moneyline-implied
  favorite on every checked game; rerun that audit any time with
  ``--sweep-dates``.
- ``p_home_implied_devig`` is the proportional de-vig of the two moneylines.
- Records append ONLY through ``common.append_log_record`` (append-only,
  hash-chained JSONL). A game with no odds anywhere still gets a record with
  null odds fields: the capture attempt itself is evidence.
- Team mapping fails closed: an ESPN abbreviation absent from
  ``common.ESPN_TO_CDN_TRICODE`` raises instead of guessing.
- Standing fence: odds captured here are for the wagering/scoring layer only,
  never inputs to ratings or projections.

Usage:
    .venv/bin/python wnba_test/capture_wnba_odds.py --date 2026-08-10
    .venv/bin/python wnba_test/capture_wnba_odds.py --date 2026-08-10 --dry-run
    .venv/bin/python wnba_test/capture_wnba_odds.py \
        --sweep-dates 2026-05-15,2026-06-05,2026-07-04,2026-08-08
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import wnba_gameday_common as common

try:  # pragma: no cover - import guard mirrors fetch_wnba_cdn
    from curl_cffi import requests as cffi_requests
except ImportError:  # pragma: no cover
    cffi_requests = None

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
EASTERN = ZoneInfo("America/New_York")
ODDS_FIELDS = ("provider", "spread_home", "home_ml", "away_ml", "over_under")
VALID_STATES = {"pre", "in", "post"}
DEFAULT_LOG_NAME = "odds_2026.jsonl"

FetchJson = Callable[[str], "dict[str, Any]"]


# --------------------------------------------------------------------------
# Fetching (network lives only here; parsing below is pure)
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
# Odds math
# --------------------------------------------------------------------------


def american_to_implied_prob(moneyline: float) -> float:
    """American moneyline -> vigged implied probability."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100.0)
    return 100.0 / (moneyline + 100.0)


def devig_home_prob(home_ml: "float | None", away_ml: "float | None") -> "float | None":
    """Proportional de-vig: p_home = q_home / (q_home + q_away)."""
    if home_ml is None or away_ml is None:
        return None
    q_home = american_to_implied_prob(home_ml)
    q_away = american_to_implied_prob(away_ml)
    total = q_home + q_away
    if total <= 0:
        return None
    return q_home / total


def _to_float(value: Any) -> "float | None":
    if value is None:
        return None
    try:
        return float(str(value).strip().replace("+", ""))
    except ValueError:
        return None


def _parse_american(value: Any) -> "int | None":
    """Parse an American moneyline that may arrive numeric or as '+142'/'EVEN'."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"OFF", "--", "N/A", "NONE"}:
        return None
    if text in {"EVEN", "EV"}:
        return 100
    parsed = _to_float(text)
    if parsed is None:
        return None
    return int(parsed)


# --------------------------------------------------------------------------
# Pure parsing (fixture-testable, no network)
# --------------------------------------------------------------------------


def extract_odds_fields(odds_obj: "dict[str, Any]") -> "dict[str, Any]":
    """Extract provider/spread/moneylines/total from one ESPN odds object.

    Handles both observed schemas: the scoreboard object (top-level ``spread``,
    string odds nested under ``moneyline.{home,away}.close.odds``) and the
    summary ``pickcenter`` object (numeric ``homeTeamOdds.moneyLine``).
    ``spread`` is the home line in both (cross-checked against
    ``pointSpread.home.close.line`` and moneyline favorites).
    """
    provider = (odds_obj.get("provider") or {}).get("name")
    spread_home = _to_float(odds_obj.get("spread"))
    if spread_home is None:
        point_spread = odds_obj.get("pointSpread") or {}
        spread_home = _to_float(
            ((point_spread.get("home") or {}).get("close") or {}).get("line")
        )
    over_under = _to_float(odds_obj.get("overUnder"))
    if over_under is None:
        total = odds_obj.get("total") or {}
        over_under = _to_float(((total.get("over") or {}).get("close") or {}).get("line"))
    home_ml = _parse_american((odds_obj.get("homeTeamOdds") or {}).get("moneyLine"))
    away_ml = _parse_american((odds_obj.get("awayTeamOdds") or {}).get("moneyLine"))
    moneyline = odds_obj.get("moneyline") or {}
    if home_ml is None:
        home_ml = _parse_american(
            ((moneyline.get("home") or {}).get("close") or {}).get("odds")
        )
    if away_ml is None:
        away_ml = _parse_american(
            ((moneyline.get("away") or {}).get("close") or {}).get("odds")
        )
    return {
        "provider": provider,
        "spread_home": spread_home,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "over_under": over_under,
    }


def _null_odds_fields() -> "dict[str, Any]":
    return {field: None for field in ODDS_FIELDS}


def _finalize_derived(record: "dict[str, Any]") -> None:
    spread_home = record.get("spread_home")
    record["market_home_margin"] = -spread_home if spread_home is not None else None
    record["p_home_implied_devig"] = devig_home_prob(
        record.get("home_ml"), record.get("away_ml")
    )


def _eastern_game_date(tipoff_utc: str) -> str:
    """ESPN event date ('2026-08-12T02:00Z') -> US/Eastern calendar date string."""
    text = str(tipoff_utc).strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return str(tipoff_utc)[:10]
    return moment.astimezone(EASTERN).date().isoformat()


def parse_scoreboard(
    payload: "dict[str, Any]", game_date: "str | None" = None
) -> "list[dict[str, Any]]":
    """Parse one ESPN scoreboard payload into odds-snapshot records (pure).

    Returns one record per event with every odds-log field except
    ``captured_at_utc`` (the caller stamps capture time). Odds fields are None
    when the scoreboard embeds no odds; the caller may fill them from the
    event summary via ``parse_pickcenter``. Unknown teams raise (fail closed).
    """
    records: "list[dict[str, Any]]" = []
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
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        home_tricode = common.espn_to_cdn_tricode(
            home_team.get("abbreviation"), home_team.get("displayName")
        )
        away_tricode = common.espn_to_cdn_tricode(
            away_team.get("abbreviation"), away_team.get("displayName")
        )
        state = ((event.get("status") or {}).get("type") or {}).get("state")
        if state not in VALID_STATES:
            raise ValueError(f"event {event.get('id')!r}: unexpected state {state!r}")
        tipoff_utc = str(event.get("date") or competition.get("date") or "")
        odds_list = competition.get("odds") or []
        fields = extract_odds_fields(odds_list[0]) if odds_list else _null_odds_fields()
        record: "dict[str, Any]" = {
            "type": "odds_snapshot",
            "espn_event_id": str(event.get("id")),
            "game_date": game_date or _eastern_game_date(tipoff_utc),
            "home_tricode": home_tricode,
            "away_tricode": away_tricode,
            "tipoff_utc": tipoff_utc,
            "state": state,
        }
        record.update(fields)
        _finalize_derived(record)
        records.append(record)
    records.sort(key=lambda row: (row["tipoff_utc"], row["espn_event_id"]))
    return records


def parse_pickcenter(summary_payload: "dict[str, Any]") -> "dict[str, Any] | None":
    """Extract odds fields from a summary payload's ``pickcenter[0]`` (pure)."""
    pickcenter = summary_payload.get("pickcenter") or []
    if not pickcenter:
        return None
    return extract_odds_fields(pickcenter[0])


# --------------------------------------------------------------------------
# Capture flow
# --------------------------------------------------------------------------


def _needs_summary(record: "dict[str, Any]") -> bool:
    return all(record.get(field) is None for field in ODDS_FIELDS)


def capture_date(
    date_str: str,
    *,
    fetch: FetchJson = fetch_json,
    payload: "dict[str, Any] | None" = None,
) -> "list[dict[str, Any]]":
    """Fetch + parse one date's odds records, filling gaps from event summaries."""
    if payload is None:
        scoreboard_url = common.ESPN_SCOREBOARD_URL.format(
            yyyymmdd=date_str.replace("-", "")
        )
        payload = fetch(scoreboard_url)
    records = parse_scoreboard(payload, game_date=date_str)
    for record in records:
        if not _needs_summary(record):
            continue
        summary_url = common.ESPN_SUMMARY_URL.format(event_id=record["espn_event_id"])
        try:
            fields = parse_pickcenter(fetch(summary_url))
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: summary fetch failed for {record['espn_event_id']}: {exc}")
            fields = None
        if fields is not None:
            record.update(fields)
            _finalize_derived(record)
    return records


def spread_ml_sign_agreement(
    records: "list[dict[str, Any]]",
) -> "tuple[int, int, list[str]]":
    """(agreements, checked, disagreement descriptions) for spread-vs-ML favorite sign.

    A record counts when it has a nonzero home spread and a de-vigged home
    probability away from 0.5; agreement means spread_home < 0 exactly when the
    moneylines make the home team the favorite.
    """
    agree = 0
    checked = 0
    disagreements: "list[str]" = []
    for record in records:
        spread_home = record.get("spread_home")
        p_home = record.get("p_home_implied_devig")
        if spread_home in (None, 0) or p_home is None or p_home == 0.5:
            continue
        checked += 1
        if (spread_home < 0) == (p_home > 0.5):
            agree += 1
        else:
            disagreements.append(
                f"{record['game_date']} {record['away_tricode']}@{record['home_tricode']}"
                f" spread_home={spread_home} p_home={p_home:.3f}"
            )
    return agree, checked, disagreements


def run_sweep(dates: "list[str]", *, fetch: FetchJson = fetch_json) -> int:
    """Alias-map + spread-semantics audit across scoreboard dates (no log writes)."""
    abbr_map: "dict[str, str]" = {}
    covered: "set[str]" = set()
    unmapped: "set[tuple[str, str]]" = set()
    all_records: "list[dict[str, Any]]" = []
    for date_str in dates:
        url = common.ESPN_SCOREBOARD_URL.format(yyyymmdd=date_str.replace("-", ""))
        payload = fetch(url)
        events = payload.get("events") or []
        date_unmapped: "set[tuple[str, str]]" = set()
        for event in events:
            for side in (event.get("competitions") or [{}])[0].get("competitors") or []:
                team = side.get("team") or {}
                abbr = str(team.get("abbreviation") or "")
                name = team.get("displayName")
                try:
                    tricode = common.espn_to_cdn_tricode(abbr, name)
                except KeyError:
                    date_unmapped.add((abbr, str(name)))
                    continue
                abbr_map[abbr] = tricode
                covered.add(tricode)
        if date_unmapped:
            unmapped.update(date_unmapped)
            print(f"{date_str}: {len(events)} events (skipping odds parse: unmapped team)")
            continue
        records = capture_date(date_str, fetch=fetch, payload=payload)
        all_records.extend(records)
        with_odds = sum(1 for row in records if not _needs_summary(row))
        print(f"{date_str}: {len(records)} events, {with_odds} with odds")
    agree, checked, disagreements = spread_ml_sign_agreement(all_records)
    expected = {
        "ATL", "CHI", "CON", "DAL", "GSV", "IND", "LAS", "LVA",
        "MIN", "NYL", "PDX", "PHX", "SEA", "TOR", "WAS",
    }
    print(f"abbreviations observed: {json.dumps(abbr_map, sort_keys=True)}")
    print(f"teams covered: {len(covered)}/15; missing: {sorted(expected - covered)}")
    print(f"spread-vs-moneyline favorite sign agreement: {agree}/{checked}")
    for line in disagreements:
        print(f"  DISAGREE {line}")
    if unmapped:
        print(f"UNMAPPED ESPN teams (extend ESPN_TO_CDN_TRICODE): {sorted(unmapped)}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Game date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=common.ODDS_LOG_DIR / DEFAULT_LOG_NAME,
        help=f"Odds log override (default: ODDS_LOG_DIR/{DEFAULT_LOG_NAME})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print records without appending to the log",
    )
    parser.add_argument(
        "--sweep-dates",
        default=None,
        help="Comma-separated YYYY-MM-DD dates: audit alias map + spread sign "
        "semantics across scoreboards, write nothing",
    )
    args = parser.parse_args()

    if args.sweep_dates:
        return run_sweep([d.strip() for d in args.sweep_dates.split(",") if d.strip()])

    datetime.strptime(args.date, "%Y-%m-%d")  # validate early
    records = capture_date(args.date)
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for record in records:
        record["captured_at_utc"] = captured_at
    if args.dry_run:
        for record in records:
            print(json.dumps(record, sort_keys=True))
        print(f"dry-run: {len(records)} records for {args.date}, nothing written")
        return 0
    for record in records:
        common.append_log_record(args.log_path, record)
    ok, n_records, reason = common.verify_log_chain(args.log_path)
    print(
        f"appended {len(records)} records for {args.date} to {args.log_path}; "
        f"chain ok={ok} n={n_records}{' ' + reason if reason else ''}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
