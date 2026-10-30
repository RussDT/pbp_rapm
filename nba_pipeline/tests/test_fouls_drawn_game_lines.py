import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import build_fouls_drawn_game_lines as fd
from build_fouls_drawn_game_lines import aggregate, build_lines, normalize, DRAWN


def foul(game, fouler, drawn, sub_type, descriptor=None, n=1):
    return {"game_id": game, "action_number": n, "order_number": n, "period": 1, "clock": "PT05M00.00S",
            "time_actual": None, "team_id": 1, "fouler_id": fouler, "fouler_name": None, "drawn_id": drawn,
            "drawn_name": None, "sub_type": sub_type, "descriptor": descriptor, "qualifiers": None,
            "official_id": 9, "description": ""}


def ledger():
    return pd.DataFrame([
        foul("g1", 10, 20, "personal", "shooting", 1),
        foul("g1", 20, 30, "offensive", "charge", 2),
        foul("g1", 10, 30, "offensive", None, 3),          # moving screen / push-off: offensive, not a charge
        foul("g1", 10, None, "technical", None, 4),        # technicals have no drawer and are not PFD
        foul("g1", 30, 10, "personal", "loose ball", 5),
        foul("g1", 30, 10, "Personal", "loose-ball", 6),   # vocabulary drift
        foul("g1", 40, 20, "offensive", "offtheball", 7),
    ])


def lines(games=("g1",), players=(10, 20, 30, 40), season=2022):
    rows = []
    for g in games:
        for p in players:
            rows.append({"game_id": g, "player_id": p, "player_name": f"P{p}", "season": season, "season_type": "rs",
                         "game_date": pd.Timestamp("2021-11-01"), "minutes": 20., "pf_committed": 0., "off_fouls_committed": 0.,
                         "charges_committed": 0., "off_poss": 50, "def_poss": 50})
    return pd.DataFrame(rows)


def test_normalize_collapses_descriptor_vocabulary():
    assert normalize("loose ball") == normalize("loose-ball") == normalize("looseball") == "looseball"
    assert normalize("Off-The-Ball") == "offtheball" and normalize(None) is None and normalize(np.nan) is None


def test_aggregate_types_fouls_and_excludes_technicals():
    drawn, committed, games = aggregate(ledger())
    d = drawn.set_index("player_id")
    assert d.loc[30, "pfd"] == 2 and d.loc[30, "off_fouls_drawn"] == 2 and d.loc[30, "charges_drawn"] == 1
    assert d.loc[30, "off_fouls_drawn_noncharge"] == 1
    assert d.loc[20, "pfd"] == 2 and d.loc[20, "shooting_fouls_drawn"] == 1 and d.loc[20, "offball_off_fouls_drawn"] == 1
    assert d.loc[10, "pfd"] == 2 and d.loc[10, "looseball_fouls_drawn"] == 2 and d.loc[10, "off_fouls_drawn"] == 0
    c = committed.set_index("player_id")
    assert c.loc[10, "pf_committed_cdn"] == 2 and c.loc[10, "off_fouls_committed_cdn"] == 1   # technical excluded
    assert c.loc[20, "charges_committed_cdn"] == 1 and c.loc[40, "off_fouls_committed_cdn"] == 1
    assert games.set_index("game_id").loc["g1", "nontechnical_fouls"] == 6
    assert games.set_index("game_id").loc["g1", "fouls_without_drawer"] == 0
    assert drawn.pfd.sum() == committed.pf_committed_cdn.sum() == 6


def test_build_lines_zero_fills_only_reconciled_games_and_marks_everything_else_unmeasured():
    led = ledger()
    led = pd.concat([led, pd.DataFrame([
        foul("g2", 10, 20, "personal", "shooting", 1),
        foul("g3", 10, 20, "personal", "shooting", 1),
        foul("g4", 10, 20, "personal", "shooting", 1),
        foul("g5", 10, None, "personal", "shooting", 1),    # personal foul with no drawer: identity fails
        foul("g6", 10, 20, "personal", "shooting", 1),
    ])], ignore_index=True)
    status = pd.DataFrame({"game_id": ["g1", "g2", "g3", "g4", "g5", "g7"], "status": ["ok", "ok", "ok", "ok", "ok", "unavailable"]})
    ln = lines(games=("g1", "g2", "g3", "g4", "g5", "g6", "g7"))
    pf = {("g1", 10): 2., ("g1", 20): 1., ("g1", 30): 2., ("g1", 40): 1., ("g2", 10): 1., ("g4", 10): 1., ("g5", 10): 1., ("g6", 10): 1.}
    ln["pf_committed"] = [pf.get((g, p), 0.) for g, p in zip(ln.game_id, ln.player_id)]
    ln.loc[ln.game_id.eq("g3") & ln.player_id.eq(10), "pf_committed"] = 2.   # PBP disagrees with the ledger
    def box(pf, pfd):
        return pd.DataFrame({"PF": pf, "PFD": pfd})
    archive = {"g1": box({10: 2, 20: 1, 30: 2, 99: 5}, {10: 2, 20: 2, 30: 2, 99: 3}),   # 99 misfiled from another game: ignored
               "g2": box({10: 1}, {20: 2}),                                                # PFD disagrees on a shared player
               "g3": box({20: 0}, {20: 1}),                                                # agrees but does not cover the active player 10
               "g4": None, "g5": None, "g6": None}
    out, checks = build_lines(led, status, ln, archive_lookup=lambda gid, season: archive.get(gid))
    reason = checks.set_index("game_id").reason
    assert reason["g1"] == "observed" and reason["g4"] == "observed"        # archive absent but PBP agrees
    assert reason["g2"] == "archive_box_mismatch" and reason["g3"] == "pbp_committed_mismatch"
    assert reason["g5"] == "drawer_identity_mismatch" and reason["g6"] == "unfetched" and reason["g7"] == "unavailable"
    assert checks.set_index("game_id").loc["g1", "archive_players_checked"] == 3
    o = out.set_index(["game_id", "player_id"])
    assert o.loc[("g1", 40), "pfd"] == 0 and o.loc[("g1", 40), "fouls_drawn_is_measured"]   # reconciled zero
    assert o.loc[("g1", 30), "off_fouls_drawn"] == 2 and o.loc[("g1", 30), "charges_drawn"] == 1
    assert o.loc[("g4", 20), "pfd"] == 1 and o.loc[("g4", 10), "pfd"] == 0
    for g in ("g2", "g3", "g5", "g6", "g7"):
        assert not o.loc[(g, 20), "fouls_drawn_is_measured"] and np.isnan(o.loc[(g, 20), "pfd"])
        assert np.isnan(o.loc[(g, 10), "off_fouls_drawn"])                                 # never zero-filled
    assert set(out.columns) >= set(DRAWN) | {"fouls_drawn_is_measured", "fouls_drawn_reason"}
    assert out.fouls_drawn_reason.eq(out.game_id.map(reason)).all()


def test_pre_2020_rows_and_duplicate_keys_are_excluded():
    led = ledger()
    ln = pd.concat([lines(season=2019), lines(season=2022)], ignore_index=True)
    ln = pd.concat([ln, ln.iloc[[4]]], ignore_index=True)     # duplicate (game, player) exposure key is ambiguous
    status = pd.DataFrame({"game_id": ["g1"], "status": ["ok"]})
    out, checks = build_lines(led, status, ln, archive_lookup=lambda gid, season: None)
    assert out.season.eq(2022).all() and len(out) == 3 and not out.duplicated(["game_id", "player_id"]).any()
    assert fd.FIRST_SEASON == 2020


def test_double_fouls_and_archive_arbitration():
    # g1: a double foul the PBP counter cannot see (player 10's last foul) is tolerated; a larger gap is not.
    led = pd.DataFrame([
        foul("g1", 10, 20, "personal", "double", 1), foul("g1", 20, 10, "personal", "double", 2),
        foul("g1", 20, 30, "personal", "shooting", 3),
        foul("g2", 10, 20, "personal", "shooting", 1), foul("g2", 10, 30, "personal", "shooting", 2),
        foul("g3", 10, 20, "personal", "shooting", 1), foul("g3", 10, 30, "personal", "shooting", 2),
    ])
    status = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "status": ["ok"]*3})
    ln = lines(games=("g1", "g2", "g3"))
    pf = {("g1", 10): 0., ("g1", 20): 2., ("g2", 10): 1., ("g3", 10): 1.}   # g2/g3: the repo PBP is missing one foul event
    ln["pf_committed"] = [pf.get((g, p), 0.) for g, p in zip(ln.game_id, ln.player_id)]
    def box(pf, pfd):
        return pd.DataFrame({"PF": pf, "PFD": pfd})
    archive = {"g1": None, "g2": box({10: 2, 20: 0, 30: 0}, {10: 0, 20: 1, 30: 1}), "g3": None}
    out, checks = build_lines(led, status, ln, archive_lookup=lambda gid, season: archive.get(gid))
    c = checks.set_index("game_id")
    assert c.loc["g1", "reason"] == "observed" and c.loc["g1", "pbp_committed_match"] and c.loc["g1", "double_fouls"] == 2
    assert c.loc["g2", "reason"] == "observed" and not c.loc["g2", "pbp_committed_match"] and c.loc["g2", "archive_covers_activity"]
    assert c.loc["g3", "reason"] == "pbp_committed_mismatch"
    o = out.set_index(["game_id", "player_id"])
    assert o.loc[("g1", 10), "pfd"] == 1 and o.loc[("g1", 20), "pfd"] == 1 and o.loc[("g1", 30), "pfd"] == 1
    assert o.loc[("g2", 20), "pfd"] == 1 and np.isnan(o.loc[("g3", 20), "pfd"])
