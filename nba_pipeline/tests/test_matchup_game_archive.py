"""Source identity, direction and missing-value gates for matchup retrieval."""
import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("matchup_archive", Path(__file__).parents[1] / "scripts/fetch_matchup_game_archive.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    stats = dict(partialPossessions=8.5, playerPoints=7, matchupFieldGoalsMade=3,
                 matchupFieldGoalsAttempted=5, matchupThreePointersMade=1,
                 matchupThreePointersAttempted=2)
    attacker = dict(personId=11, firstName="Attacker", familyName="Home",
                    matchups=[dict(personId=22, firstName="Defender", familyName="Away", statistics=stats)])
    away = dict(personId=22, firstName="Defender", familyName="Away", matchups=[])
    raw = dict(boxScoreMatchups=dict(gameId="0021700002",
        homeTeam=dict(teamId=10, players=[attacker]), awayTeam=dict(teamId=20, players=[away])))
    game = dict(game_id="0021700002", game_date=pd.Timestamp("2017-10-17"), season=2018,
                season_type="rs", home_team_id=10, away_team_id=20)
    return raw, game


def test_native_direction_missing_fields_and_fractional_exposure():
    raw, game = fixture()
    d, report = m.normalize(raw, game)
    assert d.offense_player_id.tolist() == [11]
    assert d.defense_player_id.tolist() == [22]
    assert d.offense_team_id.tolist() == [10]
    assert d.defense_team_id.tolist() == [20]
    assert d.partialPossessions.iloc[0] == 8.5
    assert "matchupFreeThrowsAttempted" not in d  # absent must not become zero
    assert report["status"] == "ok"
    assert report["tracked_fta"] is None
    assert d.source_counts_valid.all()


@pytest.mark.parametrize("bad", ["game_id", "home_team", "opponent", "duplicate"])
def test_identity_roster_and_pair_gates(bad):
    raw, game = fixture()
    box = raw["boxScoreMatchups"]
    if bad == "game_id":
        box["gameId"] = "0021700003"
    elif bad == "home_team":
        box["homeTeam"]["teamId"] = 999
    elif bad == "opponent":
        box["homeTeam"]["players"][0]["matchups"][0]["personId"] = 11
    else:
        a = box["homeTeam"]["players"][0]
        a["matchups"].append(copy.deepcopy(a["matchups"][0]))
    with pytest.raises(ValueError):
        m.normalize(raw, game)


def test_source_count_defects_retained_but_not_accepted():
    raw, game = fixture()
    raw["boxScoreMatchups"]["homeTeam"]["players"][0]["matchups"][0]["statistics"]["matchupFieldGoalsMade"] = 9
    d, report = m.normalize(raw, game)
    assert d.matchupFieldGoalsMade.iloc[0] == 9
    assert not d.source_counts_valid.any()
    assert report["makes_over_attempts"] == ["matchupFieldGoalsMade"]


def test_empty_response_is_not_a_zero_observation():
    raw, game = fixture()
    raw["boxScoreMatchups"]["homeTeam"]["players"][0]["matchups"] = []
    d, report = m.normalize(raw, game)
    assert d.empty
    assert report == dict(status="empty", rows=0)


def test_defense_only_player_absent_from_native_offensive_roster():
    raw, game = fixture()
    raw['boxScoreMatchups']['awayTeam']['players'] = []
    d, report = m.normalize(raw, game)
    assert d.defense_player_id.tolist() == [22]
    assert not d.defender_in_opposing_native_roster.any()
    assert report['defenders_absent_opposing_native_roster'] == 1


def test_uncovered_game_has_zero_inner_team_ids():
    raw, game = fixture()
    for side in ['homeTeam', 'awayTeam']:
        raw['boxScoreMatchups'][side] = dict(teamId=0, players=[])
    d, report = m.normalize(raw, game)
    assert d.empty and report['status'] == 'empty'


def test_cached_source_hash_must_match(tmp_path):
    raw, game = fixture()
    for folder in ["raw", "metadata", "pairs"]:
        (tmp_path/folder).mkdir()
    gid = game["game_id"]
    (tmp_path/"raw"/f"{gid}.json").write_text(json.dumps(raw))
    (tmp_path/"metadata"/f"{gid}.json").write_text(json.dumps(dict(status="empty", sha256="wrong")))
    with pytest.raises(ValueError, match="hash changed"):
        m.fetch_one(game, tmp_path, m.RateLimit(.25), 0)


def test_completed_pilot_replay_preserves_direction():
    path = m.ROOT / "cache/matchup_game_probe_v1/0022500004.json"
    if not path.exists():
        pytest.skip("Private live pilot is not required in source checkout")
    game = dict(game_id="0022500004", game_date=pd.Timestamp("2025-10-22"), season=2026, season_type="rs")
    d, report = m.normalize(json.loads(path.read_text()), game)
    assert len(d) == 260
    assert report["tracked_points"] == 216
    assert d.source_counts_valid.all()


def test_network_retry_replaces_failed_session_and_preserves_identity(tmp_path, monkeypatch):
    raw, game = fixture()
    game['game_id'] = '0021700999'
    raw['boxScoreMatchups']['gameId'] = game['game_id']
    for folder in ['raw', 'metadata', 'pairs']:
        (tmp_path/folder).mkdir()

    class Reply:
        status_code = 200
        content = json.dumps(raw).encode()
        def json(self):
            return raw
        def raise_for_status(self):
            return None

    class Session:
        def __init__(self, fail=False):
            self.headers = {}
            self.fail = fail
            self.closed = False
        def get(self, *args, **kwargs):
            if self.fail:
                raise m.requests.ReadTimeout('stale transport')
            return Reply()
        def close(self):
            self.closed = True

    m.reset_http_session()
    first, second = Session(True), Session(False)
    available = iter([first, second])
    monkeypatch.setattr(m.requests, 'Session', lambda: next(available))
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)
    receipt = m.fetch_one(game, tmp_path, m.RateLimit(0), 1)
    assert receipt['status'] == 'ok' and receipt['attempts'] == 2
    assert first.closed and not second.closed
    assert pd.read_parquet(tmp_path/'pairs'/'0021700999.parquet').offense_player_id.tolist() == [11]
    m.reset_http_session()


def test_service_refusal_aborts_batch_without_retry(tmp_path, monkeypatch):
    _, game = fixture()
    game['game_id'] = '0021700998'
    for folder in ['raw', 'metadata', 'pairs']:
        (tmp_path/folder).mkdir()
    class Session:
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            return type('Reply', (), {'status_code': 429})()
    session = Session()
    monkeypatch.setattr(m, 'http_session', lambda: session)
    limiter = m.RateLimit(0)
    with pytest.raises(PermissionError, match='429'):
        m.fetch_one(game, tmp_path, limiter, 3)
    assert limiter.aborted.is_set() and session.calls == 1
    with pytest.raises(PermissionError, match='stopped'):
        limiter.wait()
    assert not (tmp_path/'raw'/'0021700998.json').exists()
