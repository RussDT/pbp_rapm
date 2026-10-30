import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_estimated_deflections import forward_targets, cutoff_cells, baseline_states, subpanel, engine, paired


def panel(n=60):
    return pd.DataFrame({"player_id": [1]*n, "game_date": pd.date_range("2020-01-01", periods=n),
        "season": [2020]*n, "game_id": np.arange(n), "deflections": np.arange(n, dtype=float),
        "def_poss": [50.]*n, "source_minutes": [24.]*n, "valid": [True]*n, "full_row": np.arange(n)})


def test_forward_window_excludes_current_and_stops_at_41():
    d = panel()
    t, w = forward_targets(d)
    assert t[0] == np.arange(1, 42).sum()/(41*50)
    assert w[0] == 41*50
    d.loc[0, "deflections"] = 9999
    assert forward_targets(d)[0][0] == t[0]


def test_forward_window_cannot_cross_season_and_rejects_missing_coverage():
    d = panel()
    d.loc[20:, "season"] = 2021
    t, w = forward_targets(d)
    assert w[0] == 19*50
    assert np.isnan(t[15])
    d.loc[1:10, "valid"] = False
    assert np.isnan(forward_targets(d)[0][0])


def test_future_observations_do_not_change_past_filter_states():
    d = panel()
    params = [10000., .001, 50., .0001, .0001, 1.]
    a = engine("ou", subpanel(d), params, .03)[2]
    d.loc[40:, "deflections"] = 999
    b = engine("ou", subpanel(d), params, .03)[2]
    np.testing.assert_allclose(a[:40], b[:40])


def test_missing_rows_are_not_zero_observations():
    d = panel()
    d.loc[10, "valid"] = False
    sub = subpanel(d)
    assert 10 not in sub.full_row.values
    assert sub.loc[sub.full_row.eq(11), "days_gap"].iloc[0] == 2
    states = baseline_states(sub)
    assert states["career"][10] == (np.arange(12).sum()-10)/(11*50)


def test_cutoff_excludes_same_day_and_does_not_score_unlisted_season():
    frames = []
    for player in range(30):
        d = panel(90)
        d.player_id = player
        d.game_date = pd.date_range("2019-11-01", periods=90)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d.full_row = np.arange(len(d))
    sub = subpanel(d)
    states = {"test": sub.deflections.to_numpy()/50}
    cells = cutoff_cells(d, sub, states, [2020])
    first = cells[cells["asof"].eq(pd.Timestamp("2019-12-01"))]
    np.testing.assert_allclose(first.test, 29*2)  # Nov 30 posterior, not Dec 1
    assert set(cells.season) == {2020}


def test_paired_identical_models_have_zero_difference():
    c = pd.DataFrame({"asof": ["2024-01-01"]*30+["2024-02-01"]*30,
        "player_id": list(range(30))*2, "e": [1000.]*60, "target": [2.]*60,
        "candidate": [1.]*60, "baseline": [1.]*60})
    r = paired(c, "candidate", "baseline", draws=50)
    assert r["delta_mse"] == r["delta_se"] == r["gain_pct"] == 0
    assert r["gain_ci95"] == [0., 0.]
