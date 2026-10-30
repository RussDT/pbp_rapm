import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_deflections_next_game import next_game_cells, compare, choose
from build_estimated_deflections import subpanel, engine


def panel():
    return pd.DataFrame({"player_id": [1]*12, "player_name": ["Player"]*12,
        "game_date": pd.date_range("2020-01-01", periods=12), "season": [2020]*12,
        "game_id": [str(x) for x in range(12)], "deflections": np.arange(12, dtype=float),
        "def_poss": [50.]*12, "source_minutes": [24.]*12, "valid": [True]*12,
        "full_row": np.arange(12)})


def test_current_count_and_exposure_cannot_change_pregame_prediction():
    d = panel()
    params = [10000., .001, 50., .0001, .0001, 1.]
    a = engine("ou", subpanel(d), params, .03)
    d.loc[5, "deflections"] = 1000
    d.loc[5, "def_poss"] = 1
    b = engine("ou", subpanel(d), params, .03)
    np.testing.assert_allclose(a[0][:6], b[0][:6])
    assert a[2][5] != b[2][5]


def test_baselines_are_strictly_prior_and_target_is_current_game():
    sub = subpanel(panel())
    c = next_game_cells(sub, {"model": np.full(len(sub), .02)}, [2020])
    assert len(c) == 11
    assert c.iloc[0].target100 == 2  # one deflection / 50 possessions
    assert c.iloc[0].career == 0   # previous game's zero
    assert c.iloc[1].career == 1   # (0+1) / (50+50) * 100
    assert c.iloc[0].prior_exposure == 50
    assert c.model.eq(2).all()


def test_score_windows_and_season_fallback():
    d = panel()
    d.loc[6:, "season"] = 2021
    sub = subpanel(d)
    c = next_game_cells(sub, {"model": np.zeros(len(sub))}, [2021])
    assert c.season.eq(2021).all()
    assert c.iloc[0].season_to_date == c.iloc[0].career


def test_equal_season_weight_and_identical_cluster_control():
    c = pd.DataFrame({"player_id": [1, 2, 1, 2], "game_id": ["a", "a", "b", "b"],
        "season": [2020, 2020, 2021, 2021], "exposure": [1., 1., 100., 100.],
        "target100": [0., 0., 0., 0.], "candidate": [1., 1., 3., 3.],
        "baseline": [1., 1., 3., 3.]})
    for cluster in ["game", "player"]:
        r = compare(c, "candidate", "baseline", cluster=cluster, draws=20)
        assert r["mse_per100"] == 5
        assert r["delta_mse"] == r["delta_se"] == 0
        assert r["gain_ci95"] == [0., 0.]


def test_one_se_gate_does_not_force_new_candidate():
    assert choose({"delta_mse": .5, "delta_se": 1}, {"delta_mse": .5, "delta_se": 1}) == ("next_random_walk", "forward_incumbent")
    assert choose({"delta_mse": 2, "delta_se": 1}, {"delta_mse": 2, "delta_se": 1}) == ("next_ou", "next_ou")
