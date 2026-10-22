"""Chronology and legacy-state preservation for neutral pregame queries."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_pregame_neutral_skills import replay, compile_kernel, centers_at_queries, query_panel
from monthly_ou_filter import FastOU

PARS = [100., .01, 10., .002, .001, .25]


def run(frame, tmp_path, transient=None, shifts=None):
    n = len(frame)
    return replay(frame, PARS, .5, np.full(n, .5),
                  np.zeros(n) if shifts is None else shifts,
                  np.zeros(n) if transient is None else transient, compile_kernel(tmp_path))


def test_zero_opportunity_queries_do_not_change_future_observation_state(tmp_path):
    full = pd.DataFrame(dict(player_id=[1] * 5, day=[1, 2, 3, 7, 9], season=[2000] * 5,
                             exposure=[2., 0., 0., 3., 0.], obs_rate=[.8, np.nan, np.nan, .2, np.nan]))
    a = run(full, tmp_path)
    sparse = full.loc[full.exposure.gt(0)].reset_index(drop=True)
    b = run(sparse, tmp_path)
    np.testing.assert_allclose(a[2][[0, 3]], b[2], rtol=0, atol=1e-14)
    np.testing.assert_allclose(a[3][[0, 3]], b[3], rtol=0, atol=1e-14)
    assert a[4].tolist() == [0., 2., 2., 2., 5.]
    assert a[5].tolist() == [-1., 1., 1., 1., 7.]
    assert a[0][1] != a[0][2]  # Idle-time evolution is visible in queries.


def test_query_neutral_state_excludes_current_context_and_outcome(tmp_path):
    frame = pd.DataFrame(dict(player_id=[1] * 3, day=[1, 2, 4], season=[2000] * 3,
                              exposure=[2., 3., 4.], obs_rate=[.8, .2, .6]))
    old = run(frame, tmp_path, transient=np.array([.1, .2, .3]))
    altered = frame.copy(); altered.loc[1:, "obs_rate"] = [1., 0.]
    new = run(altered, tmp_path, transient=np.array([.1, 9., -4.]))
    np.testing.assert_array_equal(old[0][:2], new[0][:2])
    assert old[0][2] != new[0][2]


def test_same_date_observations_share_the_prior_and_batch_update(tmp_path):
    frame = pd.DataFrame(dict(player_id=[1] * 4, day=[1, 2, 2, 3], season=[2000] * 4,
                              exposure=[2., 3., 4., 0.], obs_rate=[.8, .2, .6, np.nan]))
    values = run(frame, tmp_path)
    assert values[0][1] == values[0][2]
    assert values[2][1] == values[2][2]
    assert values[4].tolist() == [0., 2., 2., 9.]
    assert values[5].tolist() == [-1., 1., 1., 2.]


def test_original_contextual_filter_post_states_reproduce(tmp_path):
    frame = pd.DataFrame(dict(player_id=[1] * 3, day=[1, 5, 100], season=[2000, 2000, 2001],
                              exposure=[2., 3., 4.], obs_rate=[.8, .2, .6]))
    frame["player_idx"] = 0; frame["game_date"] = pd.to_datetime(frame.day, unit="D")
    frame["days_gap"] = [0., 4., 95.]; frame["season_cross"] = [0., 0., 1.]
    shifts = np.array([0., 0., .02]); transient = np.array([.1, .2, .3])
    new = run(frame, tmp_path, transient=transient, shifts=shifts)
    old = FastOU(frame, tmp_path / "legacy").run(frame.obs_rate.to_numpy(), PARS, .5,
                            np.full(3, .5), shifts, transient)
    np.testing.assert_allclose(new[2], old[2], rtol=0, atol=1e-14)
    np.testing.assert_allclose(new[3], old[3], rtol=0, atol=1e-14)
    np.testing.assert_allclose(new[0], old[0] - transient, rtol=0, atol=1e-14)


def test_pure_offseason_query_does_not_commit_offseason_variance(tmp_path):
    full = pd.DataFrame(dict(player_id=[1] * 4, day=[1, 100, 104, 110], season=[2000, 2001, 2001, 2001],
                             exposure=[2., 0., 0., 4.], obs_rate=[.8, np.nan, np.nan, .6]))
    new = run(full, tmp_path, shifts=np.full(4, .02))
    old = run(full.iloc[[0, 3]].reset_index(drop=True), tmp_path, shifts=np.full(2, .02))
    np.testing.assert_allclose(new[2][[0, 3]], old[2], rtol=0, atol=1e-14)
    assert new[5].tolist() == [-1., 1., 1., 1.]


def test_query_centers_exclude_whole_current_date(tmp_path):
    from types import SimpleNamespace
    frame = pd.DataFrame(dict(season=[2000, 2000, 2001, 2001], game_date=pd.to_datetime([
        "1999-11-01", "1999-11-02", "2000-11-01", "2000-11-02"]), num=[1., 2., 30., 40.], exposure=[2., 4., 50., 60.]))
    spec = SimpleNamespace(numer="num")
    old = centers_at_queries(frame, spec)
    frame.loc[2:, "num"] = [0., 0.]
    new = centers_at_queries(frame, spec)
    assert old[2] == new[2] == .5
    assert old[3] != new[3]


def test_missing_personal_foul_counter_is_a_query_not_zero_observation():
    from types import SimpleNamespace
    frame = pd.DataFrame(dict(player_id=[1, 1, 1], game_date=pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"]),
        season=[2019] * 3, pf=[np.nan, 0., 1.], possessions=[40., 40., 40.]))
    result = query_panel(frame, SimpleNamespace(numer="pf", denom="possessions", kind="rate"))
    assert result.exposure.tolist() == [0., 40., 40.]
    assert np.isnan(result.obs_rate.iloc[0])
    assert result.obs_rate.iloc[1] == 0.


def test_terminal_score_uses_clock_order_not_high_corrected_event_number():
    from audit_pregame_target_scoreboards import terminal_scores
    raw = pd.DataFrame(dict(game_id=[1, 1], event_num=[999, 710], period=[4, 4],
                            minute_game=[47., 48.], score=["100 - 100", "110 - 111"]))
    assert terminal_scores(raw).terminal_score.iloc[0] == 221


def test_missing_terminal_score_is_not_filled_from_partial_scoreboard():
    from audit_pregame_target_scoreboards import terminal_scores
    raw = pd.DataFrame(dict(game_id=[1, 1], event_num=[1, 2], period=[4, 4],
                            minute_game=[47., 48.], score=["100 - 100", None]))
    assert pd.isna(terminal_scores(raw).terminal_score.iloc[0])
