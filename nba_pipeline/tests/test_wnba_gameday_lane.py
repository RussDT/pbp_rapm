"""Offline pytest coverage for the WNBA game-day forecast lane.

Covers the shared contract module (parsers, id helpers, tricode map, Eastern
game-date derivation, hash chain), the game-log builder's stale-non-final
refresh and date rules, invariants of the real built game-log panel (skipped
when absent), the strict-as-of minutes engine, the emitter's pinned forecast
algebra, the pure ESPN odds parser against the checked-in fixture, the
fail-closed scorer on synthetic hash-chained logs, the one-shot RTSS
confirmation guard, and the scalar Kalman filter update math. No test touches
the network or writes outside pytest tmp_path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
WNBA_TEST_DIR = ROOT / "wnba_test"
if str(WNBA_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(WNBA_TEST_DIR))

import build_wnba_game_logs as game_logs  # noqa: E402
import build_wnba_kalman_box as kalman  # noqa: E402
import capture_wnba_odds as odds  # noqa: E402
import emit_wnba_pretip_forecasts as emit  # noqa: E402
import score_wnba_forecasts as scorer  # noqa: E402
import wnba_gameday_common as common  # noqa: E402
from project_wnba_minutes import (  # noqa: E402
    TEAM_MINUTES,
    project_team_minutes,
)

SCOREBOARD_FIXTURE = (
    ROOT / "nba_pipeline" / "tests" / "fixtures" / "espn_wnba_scoreboard_sample.json"
)


# ==========================================================================
# (1) Shared contract module
# ==========================================================================


def test_parse_iso_minutes_cases() -> None:
    assert common.parse_iso_minutes("PT28M29.00S") == pytest.approx(28.0 + 29.0 / 60.0)
    assert common.parse_iso_minutes("PT35M") == 35.0
    assert common.parse_iso_minutes("PT30.00S") == 0.5
    assert common.parse_iso_minutes("PT0M0.00S") == 0.0
    assert common.parse_iso_minutes(None) == 0.0
    assert common.parse_iso_minutes("") == 0.0
    assert common.parse_iso_minutes(float("nan")) == 0.0
    assert common.parse_iso_minutes("garbage") == 0.0


def test_game_id_season_and_type_helpers() -> None:
    assert common.game_id_season("1022600001") == 2026
    assert common.game_id_season_type("1022600001") == "rs"
    assert common.game_id_season("1042300005") == 2023
    assert common.game_id_season_type("1042300005") == "ps"
    with pytest.raises(ValueError):
        common.game_id_season_type("1032600001")


def test_espn_to_cdn_tricode_known_and_fail_closed() -> None:
    assert common.espn_to_cdn_tricode("NY") == "NYL"
    assert common.espn_to_cdn_tricode("GS") == "GSV"
    assert common.espn_to_cdn_tricode("WSH") == "WAS"
    assert common.espn_to_cdn_tricode("PHO") == "PHX"
    assert common.espn_to_cdn_tricode("POR") == "PDX"
    # nickname fallback when the abbreviation is unknown
    assert common.espn_to_cdn_tricode(None, "Portland Fire") == "PDX"
    with pytest.raises(KeyError):
        common.espn_to_cdn_tricode("XXX")
    with pytest.raises(KeyError):
        common.espn_to_cdn_tricode("COOP", "TEAM COOP")


def test_eastern_game_date_conversion_dst_and_fallback() -> None:
    # 02:00Z in August is 10pm EDT (UTC-4) the previous day
    assert common.eastern_game_date("2026-08-11T02:00:00Z") == "2026-08-10"
    # 9pm+ Pacific tip crossing midnight Eastern: 04:05Z -> 00:05 EDT same UTC date
    assert common.eastern_game_date("2026-08-11T04:05:00Z") == "2026-08-11"
    # winter (EST, UTC-5): 02:00Z is 9pm the previous day
    assert common.eastern_game_date("2026-01-15T02:00:00Z") == "2026-01-14"
    # naive timestamps are assumed UTC
    assert common.eastern_game_date("2026-08-11T02:00:00") == "2026-08-10"
    # absent/unparseable -> None so callers can fall back to another source
    assert common.eastern_game_date(None) is None
    assert common.eastern_game_date("") is None
    assert common.eastern_game_date("not-a-time") is None


def test_hash_chain_append_verify_tamper_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "chain.jsonl"
    first = common.append_log_record(log, {"type": "t", "value": 1})
    assert first["prev_hash"] == common.GENESIS_HASH
    second = common.append_log_record(log, {"type": "t", "value": 2})
    assert second["prev_hash"] == first["record_hash"]
    common.append_log_record(log, {"type": "t", "value": 3})

    ok, n_records, reason = common.verify_log_chain(log)
    assert (ok, n_records, reason) == (True, 3, "")

    # records may not pre-set chain fields
    with pytest.raises(ValueError):
        common.append_log_record(log, {"value": 4, "prev_hash": "boom"})

    # tampering with an earlier record breaks verification at that line
    lines = log.read_text().splitlines()
    row = json.loads(lines[1])
    row["value"] = 99
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n")
    ok, n_records, reason = common.verify_log_chain(log)
    assert not ok
    assert "line 2" in reason
    assert n_records == 1  # records verified before the break

    # a missing log is a valid empty chain
    assert common.verify_log_chain(tmp_path / "absent.jsonl") == (True, 0, "")


# ==========================================================================
# (1b) Game-log builder: stale-non-final refresh and Eastern game_date
# ==========================================================================


def _synthetic_boxscore(game_time_utc, game_time_local) -> dict:
    def player(pid: int, minutes: str) -> dict:
        return {
            "personId": pid,
            "name": f"P{pid}",
            "position": "G",
            "starter": "1",
            "status": "ACTIVE",
            "played": "1",
            "notPlayingReason": None,
            "statistics": {"minutes": minutes, "points": "10"},
        }

    return {
        "game": {
            "gameStatus": 3,
            "gameTimeUTC": game_time_utc,
            "gameTimeLocal": game_time_local,
            "period": 4,
            "attendance": "5000",
            "homeTeam": {"teamTricode": "LVA", "teamId": "1611661319", "score": "90",
                         "players": [player(1, "PT30M0.00S")]},
            "awayTeam": {"teamTricode": "SEA", "teamId": "1611661328", "score": "80",
                         "players": [player(2, "PT28M30.00S")]},
        }
    }


def test_extract_game_date_is_eastern_with_local_fallback() -> None:
    # 9:05pm Pacific tip: arena-local date 2026-08-10, Eastern date 2026-08-11;
    # the ESPN-keyed odds/forecast join requires the Eastern date
    payload = _synthetic_boxscore("2026-08-11T04:05:00Z", "2026-08-10T21:05:00-07:00")
    game_row, player_rows = game_logs.extract_game("1022600300", payload)
    assert game_row["game_date"] == "2026-08-11"
    assert {row["game_date"] for row in player_rows} == {"2026-08-11"}
    # missing gameTimeUTC falls back to the arena-local date
    payload = _synthetic_boxscore(None, "2026-08-10T19:00:00-07:00")
    game_row, _ = game_logs.extract_game("1022600301", payload)
    assert game_row["game_date"] == "2026-08-10"


def test_resolve_final_payload_refreshes_stale_non_final(monkeypatch) -> None:
    final = {"game": {"gameStatus": 3}}
    stale = {"game": {"gameStatus": 2}}
    calls: list[tuple[str, bool]] = []

    def fake_load(game_id: str, *, fetch_missing: bool = False, refresh: bool = False):
        calls.append((game_id, refresh))
        if game_id == "gone":
            return None
        if game_id == "stale_then_final":
            return final if refresh else stale
        if game_id == "stale_forever":
            return stale
        return final

    monkeypatch.setattr(game_logs.common, "load_cached_boxscore", fake_load)
    assert game_logs.resolve_final_payload("ok", True) == (final, "final")
    assert game_logs.resolve_final_payload("gone", True) == (None, "unloadable")
    assert game_logs.resolve_final_payload("stale_then_final", True) == (
        final, "refreshed_final"
    )
    # still non-final after refetch: skipped, but stays eligible next run
    assert game_logs.resolve_final_payload("stale_forever", True) == (
        None, "non_final_skipped"
    )
    # an offline (--no-fetch) build must never attempt a refresh fetch
    calls.clear()
    assert game_logs.resolve_final_payload("stale_forever", False) == (
        None, "non_final_skipped"
    )
    assert calls == [("stale_forever", False)]


# ==========================================================================
# (2) Real built game-log panel invariants (skip when not built)
# ==========================================================================


def _load_real_panel() -> tuple:
    if not (common.GAMES_TABLE_PATH.exists() and common.GAME_LOG_PATH.exists()):
        pytest.skip("game-log panel not built (games.csv / player_game_logs.csv absent)")
    games = pd.read_csv(common.GAMES_TABLE_PATH, dtype={"game_id": str})
    logs = pd.read_csv(common.GAME_LOG_PATH, dtype={"game_id": str})
    return games, logs


def test_games_table_finals_have_both_scores() -> None:
    games, _ = _load_real_panel()
    assert (games["status"] == 3).all()
    assert games["home_score"].notna().all()
    assert games["away_score"].notna().all()
    assert (games["home_score"] > 0).all()
    assert (games["away_score"] > 0).all()
    # a final WNBA game cannot tie (overtime resolves it)
    assert (games["home_score"] != games["away_score"]).all()


def test_sampled_team_minutes_close_to_regulation_plus_ot() -> None:
    games, logs = _load_real_panel()
    period = games.set_index("game_id")["period"]
    team_minutes = (
        logs.groupby(["game_id", "team_tricode"])["minutes"].sum().reset_index()
    )
    assert len(team_minutes) == 2 * len(games)
    sampled = team_minutes.sample(n=min(50, len(team_minutes)), random_state=20260810)
    for row in sampled.itertuples(index=False):
        ot_periods = max(int(period.loc[row.game_id]) - 4, 0)
        expected = 200.0 + 25.0 * ot_periods
        assert abs(row.minutes - expected) <= 1.5, (
            f"{row.game_id} {row.team_tricode}: minutes sum {row.minutes} vs "
            f"expected {expected}"
        )


def test_dnp_rows_have_zero_minutes() -> None:
    _, logs = _load_real_panel()
    dnp = logs[logs["played"] == 0]
    assert not dnp.empty
    assert (dnp["minutes"] == 0.0).all()


# ==========================================================================
# (3) Minutes engine on a synthetic panel
# ==========================================================================

BASE_MINUTES = {101: 32.0, 102: 32.0, 103: 30.0, 104: 28.0, 105: 28.0, 106: 26.0, 107: 24.0}


def _synthetic_minutes_panel() -> pd.DataFrame:
    rows = []
    for gid, gdate in (("g1", "2026-06-01"), ("g2", "2026-06-03"), ("g3", "2026-06-05")):
        for pid, minutes in BASE_MINUTES.items():
            rows.append(
                {
                    "game_id": gid,
                    "game_date": gdate,
                    "season": 2026,
                    "team_tricode": "AAA",
                    "person_id": pid,
                    "player_name": f"Player {pid}",
                    "status": "ACTIVE",
                    "played": 1,
                    "minutes": minutes,
                }
            )
    return pd.DataFrame(rows)


def test_minutes_projection_sums_to_200_and_is_strictly_as_of() -> None:
    logs = _synthetic_minutes_panel()
    frame = project_team_minutes(logs, "AAA", "2026-06-07")
    assert abs(frame["proj_minutes"].sum() - TEAM_MINUTES) < 1e-6
    # constant per-player minutes summing to 200 must reproduce exactly
    by_pid = frame.set_index("person_id")["proj_minutes"]
    for pid, minutes in BASE_MINUTES.items():
        assert by_pid.loc[pid] == pytest.approx(minutes, abs=1e-9)

    # a game ON the as-of date itself must not influence the projection
    same_day_rows = []
    for pid in BASE_MINUTES:
        same_day_rows.append(
            {
                "game_id": "g4",
                "game_date": "2026-06-07",
                "season": 2026,
                "team_tricode": "AAA",
                "person_id": pid,
                "player_name": f"Player {pid}",
                "status": "ACTIVE",
                "played": 1,
                "minutes": 45.0 if pid == 107 else 155.0 / 6.0,
            }
        )
    polluted = pd.concat([logs, pd.DataFrame(same_day_rows)], ignore_index=True)
    frame_polluted = project_team_minutes(polluted, "AAA", "2026-06-07")
    pd.testing.assert_frame_equal(frame_polluted, frame)


def test_minutes_exclusion_redistributes_proportionally() -> None:
    logs = _synthetic_minutes_panel()
    frame = project_team_minutes(logs, "AAA", "2026-06-07", exclude_person_ids={107})
    assert 107 not in set(frame["person_id"])
    assert abs(frame["proj_minutes"].sum() - TEAM_MINUTES) < 1e-6
    remaining = {pid: m for pid, m in BASE_MINUTES.items() if pid != 107}
    scale = TEAM_MINUTES / sum(remaining.values())
    by_pid = frame.set_index("person_id")["proj_minutes"]
    for pid, minutes in remaining.items():
        assert by_pid.loc[pid] == pytest.approx(minutes * scale, abs=1e-9)

    debug = project_team_minutes(
        logs, "AAA", "2026-06-07", exclude_person_ids={107}, include_excluded=True
    )
    excluded_row = debug[debug["person_id"] == 107]
    assert len(excluded_row) == 1
    assert excluded_row["proj_minutes"].iloc[0] == 0.0
    assert excluded_row["flag"].iloc[0] == "excluded_manual"


def test_minutes_inactive_in_latest_game_is_excluded() -> None:
    logs = _synthetic_minutes_panel()
    latest_106 = (logs["game_id"] == "g3") & (logs["person_id"] == 106)
    logs.loc[latest_106, ["status", "played", "minutes"]] = ["INACTIVE", 0, 0.0]
    frame = project_team_minutes(logs, "AAA", "2026-06-06")
    assert 106 not in set(frame["person_id"])
    assert abs(frame["proj_minutes"].sum() - TEAM_MINUTES) < 1e-6
    debug = project_team_minutes(logs, "AAA", "2026-06-06", include_excluded=True)
    flag = debug.loc[debug["person_id"] == 106, "flag"].iloc[0]
    assert flag == "excluded_inactive_latest"


def test_minutes_brand_new_team_raises() -> None:
    logs = _synthetic_minutes_panel()
    with pytest.raises(ValueError):
        project_team_minutes(logs, "ZZZ", "2026-06-07")


# ==========================================================================
# (4) Forecast algebra as implemented by the emitter
# ==========================================================================


def test_team_per100_identity() -> None:
    minutes = pd.DataFrame(
        {"person_id": [1, 2, 3, 4, 5], "proj_minutes": [40.0, 40.0, 40.0, 40.0, 40.0]}
    )
    net_by_pid = pd.Series({1: 2.0, 2: 1.0, 3: -1.0, 4: 0.5})  # pid 5 unrated
    per100, share = emit.team_per100_and_share(minutes, net_by_pid)
    # pinned identity: sum(net_i * minutes_i) / 40, unrated players contribute 0
    expected = (2.0 * 40 + 1.0 * 40 + (-1.0) * 40 + 0.5 * 40 + 0.0 * 40) / 40.0
    assert per100 == pytest.approx(expected)
    assert per100 == pytest.approx(2.5)
    assert share == pytest.approx(160.0 / 200.0)


def test_normal_cdf_reference_points() -> None:
    assert emit.normal_cdf(0.0) == pytest.approx(0.5)
    assert emit.normal_cdf(1.6448536269514722) == pytest.approx(0.95, abs=1e-6)
    assert emit.normal_cdf(-1.6448536269514722) == pytest.approx(0.05, abs=1e-6)


def test_build_forecast_record_margin_and_prob_math(monkeypatch) -> None:
    home_frame = pd.DataFrame({"person_id": [1, 2, 3, 4, 5], "proj_minutes": [40.0] * 5})
    away_frame = pd.DataFrame({"person_id": [6, 7, 8, 9, 10], "proj_minutes": [40.0] * 5})

    def fake_frames(logs, assignments, tricode, game_date):
        return (home_frame if tricode == "MIN" else away_frame), "projection"

    monkeypatch.setattr(emit, "team_minutes_frame", fake_frames)
    net_by_pid = pd.Series({1: 2.0, 2: 1.0, 3: -1.0, 4: 0.5, 6: 0.4, 7: -0.4, 8: 1.0})
    params = {"hca_pts": 2.0, "sigma_margin": 12.0, "pace": 80.0, "default_params": False}
    event = {
        "espn_event_id": "e1",
        "home_tricode": "MIN",
        "away_tricode": "LVA",
        "tipoff_utc": "2026-08-10T23:00Z",
        "state": "pre",
    }
    record = emit.build_forecast_record(
        event,
        "2026-08-10",
        None,
        None,
        net_by_pid,
        params,
        {},
        Path("board.csv"),
        "2026-08-10T00:00:00+00:00",
    )
    # home per100 = 2.5, away per100 = 1.0 (pids 9/10 unrated)
    assert record["home_per100"] == pytest.approx(2.5)
    assert record["away_per100"] == pytest.approx(1.0)
    expected_margin = (2.5 - 1.0) * 80.0 / 100.0 + 2.0
    assert record["margin_mean"] == pytest.approx(expected_margin)
    assert record["margin_mean"] == pytest.approx(3.2)
    assert record["p_home_win"] == pytest.approx(
        emit.normal_cdf(expected_margin / 12.0), abs=5e-7
    )
    assert record["minutes_source"] == "projection"
    # home rated share 0.8, away rated share 0.6 -> below 0.85 threshold
    assert record["home_rated_minutes_share"] == pytest.approx(0.8)
    assert record["away_rated_minutes_share"] == pytest.approx(0.6)
    assert record["low_coverage"] is True
    assert record["type"] == "pretip_forecast"
    assert record["cdn_game_id"] is None
    # the ESPN event state must be stored honestly on every record
    assert record["espn_state"] == "pre"


def test_load_forecast_params_pace_is_strictly_prior_season(tmp_path: Path) -> None:
    params_path = tmp_path / "forecast_params.json"
    params_path.write_text(
        json.dumps(
            {
                "hca_pts": 2.0003,
                "sigma_margin": 12.6648,
                "pace_by_season": {"2024": 82.51, "2025": 81.11},
            }
        )
    )
    params_2026 = emit.load_forecast_params(2026, params_path)
    assert params_2026["pace"] == pytest.approx(81.11)
    assert params_2026["hca_pts"] == pytest.approx(2.0003)
    assert params_2026["sigma_margin"] == pytest.approx(12.6648)
    assert params_2026["default_params"] is False

    params_2025 = emit.load_forecast_params(2025, params_path)
    assert params_2025["pace"] == pytest.approx(82.51)  # never the same season
    assert params_2025["default_params"] is False

    # no season strictly before 2024 in the file -> pace falls back to default
    params_2024 = emit.load_forecast_params(2024, params_path)
    assert params_2024["pace"] == pytest.approx(emit.DEFAULT_PACE)
    assert params_2024["default_params"] is True

    missing = emit.load_forecast_params(2026, tmp_path / "absent.json")
    assert missing["default_params"] is True
    assert missing["hca_pts"] == pytest.approx(emit.DEFAULT_HCA_PTS)
    assert missing["sigma_margin"] == pytest.approx(emit.DEFAULT_SIGMA_MARGIN)


def test_team_minutes_frame_fallback_prefers_game_log_roster() -> None:
    # SEA has log rows but nobody played -> the projection refuses; the roster
    # must then come from most-recent game-log appearances, not the stale
    # team-assignment table (which keeps a traded player's FIRST team row).
    logs = pd.DataFrame(
        [
            {"game_id": "g1", "game_date": "2026-06-01", "season": 2026,
             "team_tricode": "SEA", "person_id": 301, "player_name": "A",
             "status": "ACTIVE", "played": 0, "minutes": 0.0},
            {"game_id": "g1", "game_date": "2026-06-01", "season": 2026,
             "team_tricode": "SEA", "person_id": 302, "player_name": "B",
             "status": "ACTIVE", "played": 0, "minutes": 0.0},
            # 303 most recently appeared for LVA (traded away from SEA)
            {"game_id": "g2", "game_date": "2026-06-03", "season": 2026,
             "team_tricode": "LVA", "person_id": 303, "player_name": "C",
             "status": "ACTIVE", "played": 1, "minutes": 30.0},
        ]
    )
    stale_assignments = pd.DataFrame(
        {"player_id": [301, 303, 999], "team_id": [9, 9, 9],
         "team_tricode": ["SEA", "SEA", "SEA"]}
    )
    frame, source = emit.team_minutes_frame(logs, stale_assignments, "SEA", "2026-08-10")
    assert source == "uniform_fallback"
    # game-log roster wins: 303 (traded) and 999 (stale-only) are absent
    assert set(frame["person_id"]) == {301, 302}
    assert frame["proj_minutes"].sum() == pytest.approx(emit.TEAM_MINUTES)

    # brand-new franchise with zero game-log presence -> assignments fallback
    tor_assignments = pd.DataFrame(
        {"player_id": [401, 402], "team_id": [30, 30], "team_tricode": ["TOR", "TOR"]}
    )
    frame, source = emit.team_minutes_frame(logs, tor_assignments, "TOR", "2026-08-10")
    assert source == "uniform_fallback"
    assert set(frame["person_id"]) == {401, 402}
    assert frame["proj_minutes"].sum() == pytest.approx(emit.TEAM_MINUTES)


def _emitter_flow_env(monkeypatch, tmp_path: Path) -> None:
    """Offline emit_forecasts environment: synthetic panel/board, no network."""
    rows = []
    for team, pids in (
        ("MIN", range(1, 6)), ("LVA", range(6, 11)),
        ("SEA", range(11, 16)), ("CHI", range(16, 21)),
    ):
        for i, gdate in enumerate(("2026-08-06", "2026-08-08")):
            for pid in pids:
                rows.append(
                    {"game_id": f"g-{team}-{i}", "game_date": gdate, "season": 2026,
                     "team_tricode": team, "person_id": pid, "player_name": f"P{pid}",
                     "status": "ACTIVE", "played": 1, "minutes": 40.0}
                )
    logs = pd.DataFrame(rows)
    board = pd.DataFrame({"player_id": list(range(1, 21)), "net_rapm": [0.5] * 20})
    board_file = tmp_path / "board.csv"
    board_file.write_text("player_id,net_rapm\n")
    monkeypatch.setattr(emit, "load_game_log_panel", lambda: logs)
    monkeypatch.setattr(common, "load_current_board", lambda path=None: board)
    monkeypatch.setattr(
        common, "load_team_assignments",
        lambda season: pd.DataFrame(columns=["player_id", "team_id", "team_tricode"]),
    )
    monkeypatch.setattr(
        emit, "load_forecast_params",
        lambda season, params_path=None: {
            "hca_pts": 2.0, "sigma_margin": 12.0, "pace": 80.0, "default_params": False,
        },
    )
    monkeypatch.setattr(emit, "load_cdn_game_lookup", lambda: {})
    monkeypatch.setattr(common, "CURRENT_BOARD_PATH", board_file)


def _scoreboard_payload_pre_and_live() -> dict:
    def event(eid, home, away, date_str, state):
        return {
            "id": eid,
            "date": date_str,
            "status": {"type": {"state": state, "completed": False}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home",
                         "team": {"abbreviation": home, "displayName": home}},
                        {"homeAway": "away",
                         "team": {"abbreviation": away, "displayName": away}},
                    ]
                }
            ],
        }

    return {
        "events": [
            event("401", "MIN", "LV", "2026-08-10T23:00Z", "pre"),
            event("402", "SEA", "CHI", "2026-08-11T01:00Z", "in"),
        ]
    }


def test_emitter_records_state_and_skips_live_by_default(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _emitter_flow_env(monkeypatch, tmp_path)
    payload = _scoreboard_payload_pre_and_live()
    log = tmp_path / "forecast_log" / "forecasts.jsonl"
    records = emit.emit_forecasts(
        "2026-08-10", log, dry_run=False, allow_stale=False, fetch=lambda url: payload
    )
    # default: only the 'pre' event is emitted, the 'in' event prints a skip
    assert len(records) == 1
    assert (records[0]["home_tricode"], records[0]["away_tricode"]) == ("MIN", "LVA")
    assert records[0]["espn_state"] == "pre"
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "'in'" in out and "402" in out
    ok, n_records, _ = common.verify_log_chain(log)
    assert (ok, n_records) == (True, 1)
    stored = json.loads(log.read_text().splitlines()[0])
    assert stored["espn_state"] == "pre"  # state persists into the chained record

    # --include-live override still records the state honestly
    live_log = tmp_path / "forecast_log" / "forecasts_live.jsonl"
    records = emit.emit_forecasts(
        "2026-08-10", live_log, dry_run=False, allow_stale=False,
        fetch=lambda url: payload, include_live=True,
    )
    assert [record["espn_state"] for record in records] == ["pre", "in"]


# ==========================================================================
# (5) Odds parser on the checked-in ESPN scoreboard fixture
# ==========================================================================


def test_parse_scoreboard_fixture() -> None:
    if not SCOREBOARD_FIXTURE.exists():
        pytest.skip(f"fixture missing: {SCOREBOARD_FIXTURE}")
    payload = json.loads(SCOREBOARD_FIXTURE.read_text())
    records = odds.parse_scoreboard(payload)
    assert len(records) == 2

    no_odds, with_odds = records
    # event 1: TOR @ ATL, in progress, no embedded odds
    assert no_odds["home_tricode"] == "ATL"
    assert no_odds["away_tricode"] == "TOR"
    assert no_odds["state"] == "in"
    assert no_odds["spread_home"] is None
    assert no_odds["market_home_margin"] is None
    assert no_odds["p_home_implied_devig"] is None
    # tipoff 2026-08-11T00:00Z is 2026-08-10 US/Eastern
    assert no_odds["game_date"] == "2026-08-10"

    # event 2: CHI @ SEA, pre, scoreboard odds embedded
    assert with_odds["home_tricode"] == "SEA"
    assert with_odds["away_tricode"] == "CHI"
    assert with_odds["state"] == "pre"
    assert with_odds["spread_home"] == pytest.approx(2.5)
    assert with_odds["market_home_margin"] == pytest.approx(-2.5)
    assert with_odds["home_ml"] == 124
    assert with_odds["away_ml"] == -148
    assert with_odds["over_under"] == pytest.approx(179.5)
    assert with_odds["p_home_implied_devig"] == pytest.approx(
        odds.devig_home_prob(124, -148)
    )
    assert with_odds["p_home_implied_devig"] < 0.5  # home underdog on this line
    assert with_odds["game_date"] == "2026-08-10"

    # explicit game_date overrides the Eastern derivation
    overridden = odds.parse_scoreboard(payload, game_date="2026-08-11")
    assert {row["game_date"] for row in overridden} == {"2026-08-11"}


def test_odds_math_helpers() -> None:
    assert odds.american_to_implied_prob(-150) == pytest.approx(0.6)
    assert odds.american_to_implied_prob(150) == pytest.approx(0.4)
    # proportional de-vig: q_home / (q_home + q_away)
    q_home = 160.0 / 260.0
    q_away = 100.0 / 240.0
    assert odds.devig_home_prob(-160, 140) == pytest.approx(q_home / (q_home + q_away))
    assert odds.devig_home_prob(None, 140) is None
    assert odds._parse_american("EVEN") == 100
    assert odds._parse_american("+142") == 142
    assert odds._parse_american("OFF") is None
    assert odds._parse_american(None) is None
    assert odds._eastern_game_date("2026-08-11T00:00Z") == "2026-08-10"


# ==========================================================================
# (6) Scorer on synthetic hash-chained logs
# ==========================================================================


def _write_scorer_fixture(tmp_path: Path) -> dict:
    games_path = tmp_path / "games.csv"
    forecast_log = tmp_path / "forecast_log" / "forecasts.jsonl"
    odds_log = tmp_path / "odds_log" / "odds.jsonl"
    out_dir = tmp_path / "scoring"

    games = pd.DataFrame(
        [
            # scoreable, with odds
            {"game_id": "1022600200", "game_date": "2026-08-08", "home_tricode": "MIN",
             "away_tricode": "LVA", "status": 3, "home_score": 88, "away_score": 80,
             "game_time_utc": "2026-08-09T00:00:00Z", "period": 4},
            # scoreable, no odds snapshot
            {"game_id": "1022600201", "game_date": "2026-08-08", "home_tricode": "SEA",
             "away_tricode": "CHI", "status": 3, "home_score": 78, "away_score": 84,
             "game_time_utc": "2026-08-09T02:00:00Z", "period": 4},
            # not yet final -> PENDING
            {"game_id": "1022600205", "game_date": "2026-08-09", "home_tricode": "NYL",
             "away_tricode": "IND", "status": 2, "home_score": 40, "away_score": 38,
             "game_time_utc": "2026-08-09T23:00:00Z", "period": 2},
            # duplicated final key -> REFUSED
            {"game_id": "1022600202", "game_date": "2026-08-08", "home_tricode": "ATL",
             "away_tricode": "WAS", "status": 3, "home_score": 90, "away_score": 85,
             "game_time_utc": "2026-08-08T23:00:00Z", "period": 4},
            {"game_id": "1022600203", "game_date": "2026-08-08", "home_tricode": "ATL",
             "away_tricode": "WAS", "status": 3, "home_score": 90, "away_score": 85,
             "game_time_utc": "2026-08-08T23:00:00Z", "period": 4},
        ]
    )
    games.to_csv(games_path, index=False)

    def forecast(
        game_date, home, away, margin, p_home, tipoff,
        emitted="2026-08-08T15:00:00+00:00", state="pre",
    ):
        record = {
            "type": "pretip_forecast",
            "emitted_at_utc": emitted,
            "game_date": game_date,
            "espn_event_id": "e-" + home + away,
            "cdn_game_id": None,
            "home_tricode": home,
            "away_tricode": away,
            "tipoff_utc": tipoff,
            "minutes_source": "projection",
            "low_coverage": False,
            "default_params": False,
            "margin_mean": margin,
            "margin_sigma": 12.0,
            "p_home_win": p_home,
            "model_version": "phase0_v1",
        }
        if state is not None:  # state=None simulates a legacy pre-espn_state record
            record["espn_state"] = state
        return record

    # superseded early record for MIN/LVA: last ELIGIBLE record must win
    common.append_log_record(
        forecast_log, forecast("2026-08-08", "MIN", "LVA", 99.0, 0.99, "2026-08-09T00:00:00Z")
    )
    common.append_log_record(
        forecast_log, forecast("2026-08-08", "MIN", "LVA", 6.0, 0.69, "2026-08-09T00:00:00Z")
    )
    # post-tip re-emission (tipoff 00:00Z) — the fence must exclude it even
    # though it is physically last and honestly claims state 'pre'
    common.append_log_record(
        forecast_log,
        forecast(
            "2026-08-08", "MIN", "LVA", 50.0, 0.95, "2026-08-09T00:00:00Z",
            emitted="2026-08-09T01:00:00+00:00",
        ),
    )
    common.append_log_record(
        forecast_log, forecast("2026-08-08", "SEA", "CHI", 1.0, 0.53, "2026-08-09T02:00:00Z")
    )
    # later in-progress emission (--include-live) — non-pre state must lose to
    # the earlier 'pre' record
    common.append_log_record(
        forecast_log,
        forecast("2026-08-08", "SEA", "CHI", 9.9, 0.90, "2026-08-09T02:00:00Z", state="in"),
    )
    # legacy record without espn_state: accepted but flagged
    common.append_log_record(
        forecast_log,
        forecast("2026-08-09", "NYL", "IND", 2.0, 0.56, "2026-08-09T23:00:00Z", state=None),
    )
    common.append_log_record(
        forecast_log, forecast("2026-08-08", "ATL", "WAS", 3.0, 0.60, "2026-08-08T23:00:00Z")
    )
    # absent from the games table AND older than the table horizon -> REFUSED
    common.append_log_record(
        forecast_log,
        forecast(
            "2026-07-01", "GSV", "DAL", 4.0, 0.62, "2026-07-01T23:00:00Z",
            emitted="2026-07-01T15:00:00+00:00",
        ),
    )
    # absent from the games table but recent (>= max table date - 1d) -> PENDING
    common.append_log_record(
        forecast_log,
        forecast(
            "2026-08-09", "PDX", "TOR", 1.5, 0.55, "2026-08-10T00:00:00Z",
            emitted="2026-08-09T15:00:00+00:00",
        ),
    )

    # pre snapshot for MIN/LVA; post snapshot for SEA/CHI must be ignored
    common.append_log_record(
        odds_log,
        {
            "type": "odds_snapshot",
            "captured_at_utc": "2026-08-08T16:00:00+00:00",
            "game_date": "2026-08-08",
            "espn_event_id": "e-MINLVA",
            "home_tricode": "MIN",
            "away_tricode": "LVA",
            "tipoff_utc": "2026-08-09T00:00:00Z",
            "state": "pre",
            "provider": "DraftKings",
            "spread_home": -3.5,
            "home_ml": -160,
            "away_ml": 140,
            "over_under": 165.5,
            "market_home_margin": 3.5,
            "p_home_implied_devig": odds.devig_home_prob(-160, 140),
        },
    )
    common.append_log_record(
        odds_log,
        {
            "type": "odds_snapshot",
            "captured_at_utc": "2026-08-09T05:00:00+00:00",
            "game_date": "2026-08-08",
            "espn_event_id": "e-SEACHI",
            "home_tricode": "SEA",
            "away_tricode": "CHI",
            "tipoff_utc": "2026-08-09T02:00:00Z",
            "state": "post",
            "provider": "DraftKings",
            "spread_home": -1.5,
            "home_ml": -120,
            "away_ml": 100,
            "over_under": 160.5,
            "market_home_margin": 1.5,
            "p_home_implied_devig": odds.devig_home_prob(-120, 100),
        },
    )
    return {
        "games": games_path,
        "forecast_log": forecast_log,
        "odds_log": odds_log,
        "out_dir": out_dir,
    }


def test_scorer_refusal_semantics_and_metrics(tmp_path: Path) -> None:
    paths = _write_scorer_fixture(tmp_path)
    rc = scorer.main(
        [
            "--forecast-log", str(paths["forecast_log"]),
            "--odds-log", str(paths["odds_log"]),
            "--games", str(paths["games"]),
            "--output-dir", str(paths["out_dir"]),
        ]
    )
    assert rc == 0
    report = json.loads((paths["out_dir"] / "report.json").read_text())

    counts = report["counts"]
    assert counts["unique_forecast_keys"] == 6
    assert counts["superseded_forecast_records"] == 1
    assert counts["post_tip_excluded"] == 1  # MIN/LVA post-tip re-emission
    assert counts["nonpre_state_excluded"] == 1  # SEA/CHI 'in'-state record
    assert counts["tipoff_unknown_keys"] == 2  # GSV/DAL and PDX/TOR
    assert counts["espn_state_unknown_selected"] == 1  # legacy NYL/IND record
    assert counts["pretip_fence_refused_keys"] == 0
    assert counts["scored"] == 2
    assert counts["refused"] == 2
    assert counts["pending"] == 2
    assert counts["scored_with_odds"] == 1
    assert counts["pre_odds_snapshots"] == 1  # the post-state snapshot is ignored

    reasons = {
        (row["home_tricode"], row["away_tricode"]): row["reason"]
        for row in report["refused_rows"]
    }
    assert "no matching game" in reasons[("GSV", "DAL")]  # old absent key stays REFUSED
    assert "duplicate final rows" in reasons[("ATL", "WAS")]
    assert report["pending_rows"] == [
        {"game_date": "2026-08-09", "home_tricode": "NYL", "away_tricode": "IND"},
        {"game_date": "2026-08-09", "home_tricode": "PDX", "away_tricode": "TOR"},
    ]

    metrics = report["model_metrics"]
    # scored errors: MIN 6.0 - 8 = -2.0; SEA 1.0 - (-6) = +7.0
    # (post-tip 50.0 and in-state 9.9 records were fenced out)
    assert metrics["margin_rmse"] == pytest.approx(np.sqrt((4.0 + 49.0) / 2.0), abs=1e-6)
    assert metrics["margin_mae"] == pytest.approx(4.5, abs=1e-6)
    assert metrics["margin_mean_error"] == pytest.approx(2.5, abs=1e-6)
    assert metrics["n_home_wins"] == 1
    assert metrics["win_log_loss"] is not None
    assert metrics["pit_ks_distance"] is not None
    assert metrics["crps_mean"] == pytest.approx(
        (scorer.normal_crps(6.0, 12.0, 8.0) + scorer.normal_crps(1.0, 12.0, -6.0)) / 2.0,
        abs=1e-6,
    )

    market = report["market_comparison"]
    assert market["n_with_market_margin"] == 1
    assert market["model_margin_rmse_odds_rows"] == pytest.approx(2.0, abs=1e-6)
    assert market["market_margin_rmse"] == pytest.approx(4.5, abs=1e-6)
    # close CRPS: market_home_margin 3.5 with the MODEL's sigma, outcome space
    assert market["model_crps_odds_rows"] == pytest.approx(
        scorer.normal_crps(6.0, 12.0, 8.0), abs=1e-6
    )
    assert market["market_close_crps"] == pytest.approx(
        scorer.normal_crps(3.5, 12.0, 8.0), abs=1e-6
    )
    sim = market["spread_simulation"]
    # edge 6.0 - 3.5 = 2.5 >= 2.0 -> home bet; actual 8 > 3.5 -> win
    assert sim["n_bets"] == 1
    assert sim["record"] == "1-0-0"
    assert sim["hit_rate"] == pytest.approx(1.0)

    csv_rows = pd.read_csv(paths["out_dir"] / "scored_forecasts.csv")
    assert len(csv_rows) == 6
    assert sorted(csv_rows["verdict"]) == [
        "PENDING", "PENDING", "REFUSED", "REFUSED", "SCORED", "SCORED",
    ]
    scored_a = csv_rows[
        (csv_rows["home_tricode"] == "MIN") & (csv_rows["verdict"] == "SCORED")
    ].iloc[0]
    assert scored_a["margin_mean"] == pytest.approx(6.0)  # last PRE-TIP record won
    assert scored_a["margin_error"] == pytest.approx(6.0 - 8.0)
    assert scored_a["game_id"] == 1022600200
    assert scored_a["espn_state"] == "pre"
    assert scored_a["crps"] == pytest.approx(scorer.normal_crps(6.0, 12.0, 8.0), abs=1e-6)
    scored_b = csv_rows[
        (csv_rows["home_tricode"] == "SEA") & (csv_rows["verdict"] == "SCORED")
    ].iloc[0]
    assert scored_b["margin_mean"] == pytest.approx(1.0)  # 'in'-state record lost


def test_scorer_hard_refuses_tampered_chain(tmp_path: Path) -> None:
    paths = _write_scorer_fixture(tmp_path)
    # tamper with the second forecast record without recomputing hashes
    lines = paths["forecast_log"].read_text().splitlines()
    row = json.loads(lines[1])
    row["margin_mean"] = -50.0
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    paths["forecast_log"].write_text("\n".join(lines) + "\n")

    out_dir = tmp_path / "scoring_tampered"
    rc = scorer.main(
        [
            "--forecast-log", str(paths["forecast_log"]),
            "--odds-log", str(paths["odds_log"]),
            "--games", str(paths["games"]),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 2
    assert not (out_dir / "report.json").exists()
    assert not (out_dir / "scored_forecasts.csv").exists()


def test_scorer_empty_logs_clean_report(tmp_path: Path) -> None:
    games_path = tmp_path / "games.csv"
    pd.DataFrame(
        columns=[
            "game_id", "game_date", "home_tricode", "away_tricode",
            "status", "home_score", "away_score", "game_time_utc", "period",
        ]
    ).to_csv(games_path, index=False)
    out_dir = tmp_path / "scoring_empty"
    rc = scorer.main(
        [
            "--forecast-log", str(tmp_path / "no_forecasts.jsonl"),
            "--odds-log", str(tmp_path / "no_odds.jsonl"),
            "--games", str(games_path),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "report.json").read_text())
    assert report["counts"]["scored"] == 0
    assert report["model_metrics"]["margin_rmse"] is None
    assert any("0 scoreable" in note for note in report["notes"])


def _single_final_games_csv(tmp_path: Path) -> Path:
    games_path = tmp_path / "games.csv"
    pd.DataFrame(
        [
            {"game_id": "1022600300", "game_date": "2026-08-08", "home_tricode": "MIN",
             "away_tricode": "LVA", "status": 3, "home_score": 88, "away_score": 80,
             "game_time_utc": "2026-08-09T00:00:00Z", "period": 4},
        ]
    ).to_csv(games_path, index=False)
    return games_path


def _minimal_forecast(**extra) -> dict:
    record = {
        "type": "pretip_forecast",
        "emitted_at_utc": "2026-08-08T15:00:00+00:00",
        "game_date": "2026-08-08",
        "home_tricode": "MIN",
        "away_tricode": "LVA",
        "tipoff_utc": "2026-08-09T00:00:00Z",
        "espn_state": "pre",
        "margin_mean": 6.0,
        "margin_sigma": 12.0,
        "p_home_win": 0.69,
        "model_version": "phase0_v1",
    }
    record.update(extra)
    return record


def test_scorer_pretip_fence_refuses_key_with_only_post_tip_records(
    tmp_path: Path,
) -> None:
    games_path = _single_final_games_csv(tmp_path)
    forecast_log = tmp_path / "forecasts.jsonl"
    # emitted exactly AT tipoff (00:00Z from the games table): the fence is
    # STRICT, so this record is not pre-tip and the key must surface REFUSED
    common.append_log_record(
        forecast_log, _minimal_forecast(emitted_at_utc="2026-08-09T00:00:00+00:00")
    )
    out_dir = tmp_path / "scoring"
    rc = scorer.main(
        [
            "--forecast-log", str(forecast_log),
            "--odds-log", str(tmp_path / "no_odds.jsonl"),
            "--games", str(games_path),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "report.json").read_text())
    assert report["counts"]["post_tip_excluded"] == 1
    assert report["counts"]["pretip_fence_refused_keys"] == 1
    assert report["counts"]["scored"] == 0
    assert report["counts"]["refused"] == 1
    assert "pre-tip fence" in report["refused_rows"][0]["reason"]
    csv_rows = pd.read_csv(out_dir / "scored_forecasts.csv")
    assert list(csv_rows["verdict"]) == ["REFUSED"]


def test_scorer_ignores_unknown_extra_fields(tmp_path: Path) -> None:
    games_path = _single_final_games_csv(tmp_path)
    forecast_log = tmp_path / "forecasts.jsonl"
    # forward-compatible schema: unknown fields (e.g. future execution
    # telemetry) ride along in the hash chain and scoring must ignore them
    common.append_log_record(
        forecast_log,
        _minimal_forecast(future_extra_field="ignore-me", another_unknown=7),
    )
    out_dir = tmp_path / "scoring"
    rc = scorer.main(
        [
            "--forecast-log", str(forecast_log),
            "--odds-log", str(tmp_path / "no_odds.jsonl"),
            "--games", str(games_path),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "report.json").read_text())
    assert report["counts"]["scored"] == 1
    assert report["counts"]["refused"] == 0
    csv_rows = pd.read_csv(out_dir / "scored_forecasts.csv")
    assert csv_rows.iloc[0]["verdict"] == "SCORED"
    assert csv_rows.iloc[0]["margin_error"] == pytest.approx(6.0 - 8.0)


def test_normal_crps_matches_hand_computed_values() -> None:
    # z=0: CRPS = sigma * (2*phi(0) - 1/sqrt(pi)) = sigma * 0.2336949772551...
    assert scorer.normal_crps(0.0, 1.0, 0.0) == pytest.approx(0.2336949772551, abs=1e-12)
    assert scorer.normal_crps(5.0, 10.0, 5.0) == pytest.approx(2.336949772551, abs=1e-11)
    # z=1: 1*(2*Phi(1)-1) + 2*phi(1) - 1/sqrt(pi)
    #    = 0.6826894921 + 0.4839414490 - 0.5641895835 = 0.6024413576276
    assert scorer.normal_crps(0.0, 1.0, 1.0) == pytest.approx(0.6024413576276, abs=1e-12)
    # translation/scale: 5 * f(1.4), symmetric in the sign of the error
    assert scorer.normal_crps(10.0, 5.0, 3.0) == pytest.approx(4.545733509346, abs=1e-11)
    assert scorer.normal_crps(10.0, 5.0, 17.0) == pytest.approx(
        scorer.normal_crps(10.0, 5.0, 3.0), abs=1e-12
    )
    # sigma -> 0 limit recovers absolute error
    assert scorer.normal_crps(0.0, 1e-9, 3.0) == pytest.approx(3.0, abs=1e-6)


# ==========================================================================
# (7) One-shot RTSS confirmation guard (subprocess, offline)
# ==========================================================================

CONFIRMATION_SCRIPT = WNBA_TEST_DIR / "run_wnba_rtss_confirmation.py"
CONFIRMATION_SENTINEL = WNBA_TEST_DIR / "Results" / "rtss_confirmation" / "FIRED"


def test_confirmation_check_is_blocked_without_operator_assertion() -> None:
    result = subprocess.run(
        [sys.executable, str(CONFIRMATION_SCRIPT), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    # season_complete requires the explicit --season-complete operator
    # assertion, so --check can never pass from this suite
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BLOCKED" in result.stdout
    assert "season_complete" in result.stdout


def test_confirmation_real_mode_refuses_and_leaves_no_sentinel() -> None:
    result = subprocess.run(
        [sys.executable, str(CONFIRMATION_SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "REFUSING" in result.stdout
    assert not CONFIRMATION_SENTINEL.exists()


# ==========================================================================
# (8) Kalman scalar filter update math
# ==========================================================================


def test_kalman_filter_two_observation_hand_example() -> None:
    # player 0: two observations with a drift step between them;
    # player 1: one observation, interleaved to exercise the step layout
    df = pd.DataFrame(
        {
            "player_idx": [0, 1, 0],
            "minutes": [20.0, 10.0, 25.0],
            "days_gap": [0.0, 0.0, 3.0],
            "season_cross": [0.0, 0.0, 1.0],
            "step_k": [0, 0, 1],
        }
    )
    layout = kalman.FilterLayout(df)
    obs = np.array([1.0, 0.4, 0.5])
    q_game, q_day, q_off, r, v0 = 0.01, 0.001, 0.02, 2.0, 0.25
    mean_rate = 0.8
    pred, post, post_var = kalman.run_filter(
        layout, obs, (q_game, q_day, q_off, r, v0), mean_rate
    )

    # player 0, obs 1 (hand-computed): obs_var = 2/20 = 0.1,
    # gain = 0.25 / 0.35 = 5/7, post = 0.8 + (5/7)(1.0 - 0.8) = 33/35,
    # post_var = (2/7)(0.25) = 1/14
    assert pred[0] == mean_rate
    assert post[0] == pytest.approx(33.0 / 35.0)
    assert post_var[0] == pytest.approx(1.0 / 14.0)

    # player 1, only obs (hand-computed): obs_var = 2/10 = 0.2,
    # gain = 0.25 / 0.45 = 5/9, post = 0.8 + (5/9)(0.4 - 0.8) = 26/45
    assert pred[1] == mean_rate
    assert post[1] == pytest.approx(26.0 / 45.0)
    assert post_var[1] == pytest.approx((4.0 / 9.0) * 0.25)

    # player 0, obs 2: prior variance drifts by q_game + 3*q_day + q_offseason
    prior_var = 1.0 / 14.0 + q_game + 3.0 * q_day + q_off
    obs_var = r / 25.0
    gain = prior_var / (prior_var + obs_var)
    assert pred[2] == pytest.approx(33.0 / 35.0)  # one-step-ahead = prior posterior
    assert post[2] == pytest.approx(33.0 / 35.0 + gain * (0.5 - 33.0 / 35.0))
    assert post_var[2] == pytest.approx((1.0 - gain) * prior_var)
    # observing must reduce variance relative to the drifted prior
    assert post_var[2] < prior_var


def test_kalman_current_states_are_sane_when_built() -> None:
    states_path = common.GAMEDAY_DIR / "kalman_box" / "current_states.csv"
    if not states_path.exists():
        pytest.skip("kalman current_states.csv not built")
    states = pd.read_csv(states_path)
    rate_cols = [f"{stat}_rate" for stat in kalman.STATS if f"{stat}_rate" in states]
    assert rate_cols, "no rate columns found in current_states.csv"
    for col in rate_cols:
        assert np.isfinite(states[col]).all(), f"{col} has non-finite values"
        assert (states[f"{col.replace('_rate', '')}_var"] > 0).all()
    if "fg_pct_hat" in states:
        fg = states["fg_pct_hat"].dropna()
        assert ((fg >= 0.0) & (fg <= 1.0)).all()
    if "ft_pct_hat" in states:
        ft = states["ft_pct_hat"].dropna()
        assert ((ft >= 0.0) & (ft <= 1.0)).all()
