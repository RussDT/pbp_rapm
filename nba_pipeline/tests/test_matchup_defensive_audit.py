import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a = load("audit_matchup_defensive_features")
b = load("build_matchup_defensive_features")


def test_independent_snapshot_checks_units_and_population_shrinkage():
    # Same-day observations: weights cancel in population, but retain their
    # exact contribution to each player's prior-versus-observation precision.
    atoms = pd.DataFrame(dict(player_id=[11, 22], game_date=pd.to_datetime(["2021-01-01"] * 2),
        matchup_assignment_bpm__numerator=[20., 80.], matchup_assignment_bpm__denominator=[10., 20.]))
    cut = pd.Timestamp("2021-01-02")
    w = 2 ** (-1 / 550)
    expected = (np.array([20., 80.]) * w + 1000 * (100 / 30)) / (np.array([10., 20.]) * w + 1000)
    board = pd.DataFrame(dict(player_id=[11, 22], matchup_assignment_bpm=expected))
    assert a.independent_snapshot_values(atoms, board, "matchup_assignment_bpm", cut) < 1e-12
    board["matchup_assignment_bpm"] *= 100
    with pytest.raises(AssertionError):
        a.independent_snapshot_values(atoms, board, "matchup_assignment_bpm", cut)


def test_zero_coverage_stays_missing_even_with_population_evidence():
    atoms = pd.DataFrame(dict(player_id=[11, 22], game_date=pd.to_datetime(["2021-01-01"] * 2),
        matchup_assignment_bpm__numerator=[0., 80.], matchup_assignment_bpm__denominator=[0., 20.]))
    board = pd.DataFrame(dict(player_id=[11], matchup_assignment_bpm=[np.nan]))
    a.independent_snapshot_values(atoms, board, "matchup_assignment_bpm", pd.Timestamp("2021-01-02"))
    board["matchup_assignment_bpm"] = 4.
    with pytest.raises(AssertionError):
        a.independent_snapshot_values(atoms, board, "matchup_assignment_bpm", pd.Timestamp("2021-01-02"))


def test_source_coverage_is_observed_ratio_without_population_pseudocounts():
    atoms = pd.DataFrame(dict(player_id=[11, 22], game_date=pd.to_datetime(["2021-01-01"] * 2),
        matchup_coverage__numerator=[0., 80.], matchup_coverage__denominator=[100., 100.]))
    board = pd.DataFrame(dict(player_id=[11, 22], matchup_coverage=[0., .8]))
    a.independent_snapshot_values(atoms, board, "matchup_coverage", pd.Timestamp("2021-01-02"))


def test_same_day_and_future_opponent_results_cannot_change_earlier_price():
    base = dict(player_id=11, off_poss=100., rim_fgm=3., rim_fga=5., mid_fgm=1., mid_fga=2.,
        three_fgm=2., three_fga=4., ftm=2., fta=2., fga=11., ast=4., tov=2.)
    lines = pd.DataFrame([dict(base, game_id=i, game_date=f"2021-01-0{i}") for i in [1, 2, 3]])
    requests = pd.DataFrame(dict(offense_player_id=[11], game_date=["2021-01-02"]))
    original = b.previous_day_evidence(lines, requests)
    changed = lines.copy()
    columns = [c for c in base if c != "player_id"]
    changed.loc[changed.game_date.ge("2021-01-02"), columns] = 1e12
    mutated = b.previous_day_evidence(changed, requests)
    pd.testing.assert_frame_equal(original, mutated)
    assert original.player_evidence_day.iloc[0] == pd.Timestamp("2021-01-01")
    assert original.expected_total_points.iloc[0] == pytest.approx(.16)


def test_previous_season_metadata_and_minutes_weighted_position_mean():
    table = pd.DataFrame(dict(nba_id=[11, 22, 11], year=[2021, 2021, 2022],
        bpm=[2., 8., 999.], obpm=[3., 9., 999.], position=["PG", "PG", "C"],
        minutes=[100., 300., 1000.], source_last_game_date=pd.to_datetime(["2021-05-16", "2021-05-16", "2022-04-10"])))
    requests = pd.DataFrame(dict(offense_player_id=[11], season=[2022], game_date=["2021-10-20"]))
    original = b.lagged_metadata(requests, table)
    assert original.opponent_bpm.iloc[0] == 2
    assert original.opponent_obpm.iloc[0] == 3
    assert original.opponent_position.iloc[0] == "PG"
    assert original.opponent_bpm_position_mean.iloc[0] == 6.5
    changed = table.copy()
    changed.loc[changed.year.eq(2022), ["bpm", "obpm"]] = 1e12
    pd.testing.assert_frame_equal(original, b.lagged_metadata(requests, changed))


def test_metadata_effective_date_cannot_reach_matchup_day():
    table = pd.DataFrame(dict(nba_id=[11], year=[2021], bpm=[2.], obpm=[3.], position=["PG"], minutes=[100.],
        source_last_game_date=pd.to_datetime(["2021-10-20"])))
    requests = pd.DataFrame(dict(offense_player_id=[11], season=[2022], game_date=["2021-10-20"]))
    with pytest.raises(ValueError, match="metadata reaches"):
        b.lagged_metadata(requests, table)


def test_bpm_publication_floor_does_not_erase_known_prior_position():
    table = pd.DataFrame(dict(nba_id=[11], year=[2019], bpm=[2.], obpm=[3.], position=["PG"], minutes=[100.],
        source_last_game_date=pd.to_datetime(["2019-04-10"])))
    requests = pd.DataFrame(dict(offense_player_id=[11, 11], season=[2020, 2020],
        game_date=["2020-02-01", "2020-03-02"]))
    result = b.lagged_metadata(requests, table).sort_values("game_date")
    assert len(result) == 2 and result.opponent_bpm.isna().tolist() == [True, False]
    assert result.opponent_obpm.isna().tolist() == [True, False]
    assert result.opponent_position.eq("PG").all()
