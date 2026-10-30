import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import build_tracking_next_game as tng
from build_tracking_next_game import STATS, build_panel, build_fouls_panel, family_stats, subpanel, engine, next_game_cells, select


def tracking_frame(n=12, season=2020):
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({"PLAYER_ID": [1]*n, "PLAYER_NAME": ["Player"]*n, "TEAM_ID": [10]*n,
        "date": dates.strftime("%Y%m%d").astype(int), "MIN": [24.]*n, "AST": [3]*n, "FGA": [8]*n,
        "season": [season]*n, "playoffs": [False]*n,
        "potential_ast_observed": np.arange(n, dtype=float), "potential_ast_is_measured": [True]*n,
        "DRIVES": [5.]*n, "TIME_OF_POSS": [2.]*n, "hustle_CONTESTED_SHOTS": [6.]*n,
        "dfga_observed": [9.]*n, "dfga_is_measured": [True]*n, "DREB_CONTEST": [1.]*n})


def lines_frame(n=12, season=2020):
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({"player_id": [1]*n, "season": [season]*n, "season_type": ["rs"]*n, "game_date": dates,
        "game_id": [f"g{i}" for i in range(n)], "minutes": [24.]*n, "ast": [3]*n, "fga": [8]*n,
        "off_poss": [50]*n, "def_poss": [40]*n})


def fouls_frame(n=12, season=2021):
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({"player_id": [1]*n, "player_name": ["Player"]*n, "game_id": [f"g{i}" for i in range(n)],
        "season": [season]*n, "season_type": ["rs"]*n, "game_date": dates, "minutes": [24.]*n,
        "pfd": [3.]*n, "off_fouls_drawn": [1.]*n, "charges_drawn": [0.]*n,
        "fouls_drawn_is_measured": [True]*n, "fouls_drawn_reason": ["observed"]*n})


def test_registry_covers_the_six_unmodeled_stats_with_the_right_exposure():
    assert set(family_stats("tracking")) == {"potential_assists", "drives", "possession_minutes", "contested_shots", "dfga", "contested_dreb"}
    assert set(family_stats("fouls_drawn")) == {"off_fouls_drawn", "charges_drawn", "pfd"}
    assert set(STATS) == set(family_stats("tracking")) | set(family_stats("fouls_drawn"))
    for s in family_stats("tracking").values():
        assert s.warmup == 2019 and tuple(s.development) == (2020, 2021, 2022)
    for s in family_stats("fouls_drawn").values():
        assert s.warmup == 2020 and tuple(s.development) == (2021, 2022) and s.measured == "fouls_drawn_is_measured"
    assert STATS["off_fouls_drawn"].exposure == "def_poss" and STATS["charges_drawn"].exposure == "def_poss"
    assert STATS["pfd"].exposure == "off_poss"
    assert tng.FAMILY_OUT["fouls_drawn"] != tng.FAMILY_OUT["tracking"]
    assert {s.exposure for k, s in STATS.items() if k in ("potential_assists", "drives", "possession_minutes")} == {"off_poss"}
    assert {s.exposure for k, s in STATS.items() if k in ("contested_shots", "dfga", "contested_dreb")} == {"def_poss"}
    assert STATS["potential_assists"].measured == "potential_ast_is_measured"
    assert STATS["dfga"].measured == "dfga_is_measured"
    source = Path(tng.__file__).read_text()
    for repaired in ["potential_ast_filled", "dfga_filled", "potential_ast_estimate", "dfga_estimate"]:
        assert repaired not in source   # repaired values are never observations


def test_measured_only_mask_and_filled_columns_are_never_read():
    t = tracking_frame()
    t.loc[3, "potential_ast_is_measured"] = False   # preserved outlier row: still not an observation
    t.loc[4, "dfga_is_measured"] = False
    t.loc[5, "DRIVES"] = np.nan                    # missing tracking family: unscored, not zero
    panel, ambiguous = build_panel(t, lines_frame())   # no *_filled / *_estimate columns exist at all
    assert ambiguous == 0
    assert not panel.loc[3, "potential_assists_valid"] and np.isnan(panel.loc[3, "potential_assists_value"])
    assert panel.loc[3, "potential_assists_reason"] == "unmeasured"
    assert not panel.loc[4, "dfga_valid"] and panel.loc[4, "drives_valid"]
    assert not panel.loc[5, "drives_valid"] and panel.loc[5, "potential_assists_valid"]
    assert panel.potential_assists_valid.sum() == 11 and panel.dfga_valid.sum() == 11 and panel.drives_valid.sum() == 11
    sub = subpanel(panel, STATS["potential_assists"])
    assert len(sub) == 11 and list(sub.step_k) == list(range(11))
    assert sub.loc[sub.game_date.eq(pd.Timestamp("2020-01-05")), "days_gap"].item() == 2


def test_exposure_join_requires_box_agreement_and_unique_keys():
    t = tracking_frame()
    p = lines_frame()
    p.loc[2, "minutes"] = 26.       # minutes disagree by more than 0.5
    p.loc[7, "ast"] = 4             # assists disagree
    p = pd.concat([p, p.iloc[[9]]], ignore_index=True)   # duplicate exposure key is ambiguous
    p = p.drop(index=10).reset_index(drop=True)          # missing exposure row
    panel, ambiguous = build_panel(t, p)
    assert ambiguous == 2
    for stat in family_stats("tracking"):
        assert not panel.loc[2, stat+"_valid"] and panel.loc[2, stat+"_reason"] == "box_or_minutes_mismatch"
        assert not panel.loc[7, stat+"_valid"] and panel.loc[7, stat+"_reason"] == "box_or_minutes_mismatch"
        assert not panel.loc[9, stat+"_valid"] and panel.loc[9, stat+"_reason"] == "missing_or_ambiguous_exposure"
        assert not panel.loc[10, stat+"_valid"] and panel.loc[10, stat+"_reason"] == "missing_or_ambiguous_exposure"
        assert panel[stat+"_valid"].sum() == 8
    assert panel.loc[~panel.exposure_match, ["off_poss", "def_poss"]].isna().all().all()


def test_current_count_and_exposure_cannot_change_pregame_prediction():
    t = tracking_frame()
    panel, _ = build_panel(t, lines_frame())
    stat = STATS["potential_assists"]
    params = [10000., .001, 50., .0001, .0001, 1.]
    a = engine("ou", subpanel(panel, stat), params, .06)
    t.loc[5, "potential_ast_observed"] = 1000
    p = lines_frame()
    p.loc[5, "off_poss"] = 1
    panel2, _ = build_panel(t, p)
    b = engine("ou", subpanel(panel2, stat), params, .06)
    np.testing.assert_allclose(a[0][:6], b[0][:6])
    assert a[2][5] != b[2][5]
    rw = engine("random_walk", subpanel(panel, stat), [1e-4, 0., 0., 1., .01], .06)
    assert rw[0].shape == (12,)


def test_baselines_are_strictly_prior_and_target_is_current_game():
    panel, _ = build_panel(tracking_frame(), lines_frame())
    stat = STATS["potential_assists"]
    sub = subpanel(panel, stat)
    c = next_game_cells(sub, stat, {"model": np.full(len(sub), .02)}, [2020])
    assert len(c) == 11
    assert c.iloc[0].target100 == 2          # one potential assist / 50 offensive possessions
    assert c.iloc[0].career == 0             # previous game's zero
    assert c.iloc[1].career == 1             # (0+1) / (50+50) * 100
    assert c.iloc[0].prior_exposure == 50
    assert c.iloc[0].trail10 == 0 and c.iloc[10].trail10 == pytest.approx(100*sum(range(1, 11))/500)  # games 1..10
    assert c.model.eq(2).all()


def test_season_fallback_and_protected_seasons_cannot_be_scored():
    t = tracking_frame()
    t.loc[6:, "season"] = 2021
    p = lines_frame()
    p.loc[6:, "season"] = 2021
    panel, _ = build_panel(t, p)
    stat = STATS["drives"]
    sub = subpanel(panel, stat)
    c = next_game_cells(sub, stat, {"model": np.zeros(len(sub))}, [2021])
    assert c.season.eq(2021).all() and len(c) == 6
    assert c.iloc[0].season_to_date == c.iloc[0].career
    assert sub.season_cross.sum() == 1
    for seasons in ([2026], [2024, 2025, 2026], [2027]):
        with pytest.raises(ValueError):
            next_game_cells(sub, stat, {"model": np.zeros(len(sub))}, seasons)
    assert tng.EXPORT_ONLY == 2026 and max(tng.STRESS) == 2025 and max(tng.DEVELOPMENT) == 2022
    assert all(max(s.development) < min(tng.STRESS) for s in STATS.values())


def test_fouls_panel_keeps_reconciled_zeros_and_never_scores_unmeasured_games():
    f = fouls_frame()
    f.loc[2, ["pfd", "off_fouls_drawn", "charges_drawn"]] = 0.        # reconciled game, drew nothing: a real zero
    f.loc[3, ["pfd", "off_fouls_drawn", "charges_drawn"]] = np.nan    # unfetched / failed reconciliation
    f.loc[3, ["fouls_drawn_is_measured", "fouls_drawn_reason"]] = [False, "archive_pfd_mismatch"]
    f.loc[4, ["pfd", "off_fouls_drawn", "charges_drawn"]] = 0.        # a zero on an unmeasured game must not count
    f.loc[4, ["fouls_drawn_is_measured", "fouls_drawn_reason"]] = [False, "unavailable"]
    p = lines_frame(season=2021)
    p.loc[5, "minutes"] = 30.                                          # exposure row disagrees with the fouls line
    p = pd.concat([p, p.iloc[[6]]], ignore_index=True)                 # duplicate player-game exposure key
    p.loc[7, "def_poss"] = 0
    panel, ambiguous = build_fouls_panel(f, p)
    assert ambiguous == 2
    assert panel.loc[2, "off_fouls_drawn_valid"] and panel.loc[2, "off_fouls_drawn_value"] == 0
    assert not panel.loc[3, "pfd_valid"] and np.isnan(panel.loc[3, "pfd_value"]) and panel.loc[3, "pfd_reason"] == "archive_pfd_mismatch"
    assert not panel.loc[4, "charges_drawn_valid"] and np.isnan(panel.loc[4, "charges_drawn_value"]) and panel.loc[4, "charges_drawn_reason"] == "unavailable"
    assert not panel.loc[5, "pfd_valid"] and panel.loc[5, "pfd_reason"] == "box_or_minutes_mismatch"
    assert not panel.loc[6, "pfd_valid"] and panel.loc[6, "pfd_reason"] == "missing_or_ambiguous_exposure"
    assert not panel.loc[7, "off_fouls_drawn_valid"] and panel.loc[7, "off_fouls_drawn_reason"] == "zero_exposure"
    assert panel.loc[7, "pfd_valid"]                                   # offensive exposure is fine on that row
    assert panel.pfd_valid.sum() == 8 and panel.off_fouls_drawn_valid.sum() == 7
    sub = subpanel(panel, STATS["off_fouls_drawn"])
    assert (100*sub.obs_rate).iloc[0] == 2.5                           # one offensive foul drawn per 40 defensive possessions
    assert "source_minutes" in panel and "exposure_match" in panel


def test_possession_minutes_is_time_per_100_offensive_possessions():
    panel, _ = build_panel(tracking_frame(), lines_frame())
    sub = subpanel(panel, STATS["possession_minutes"])
    assert (100*sub.obs_rate).eq(4).all()    # 2 minutes of possession per 50 offensive possessions


def test_one_se_gate_does_not_force_ou():
    assert select({"delta_mse": .5, "delta_se": 1}) == "next_random_walk"
    assert select({"delta_mse": 2, "delta_se": 1}) == "next_ou"
    assert select({"delta_mse": -2, "delta_se": 1}) == "next_random_walk"
