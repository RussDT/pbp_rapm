"""Temporal safety of the FT target adapter against real frozen own models."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_vpm_dated_ft_context as adapter


def test_real_pregame_ft_predictions_are_prefix_invariant_and_exclude_same_day(monkeypatch):
    model_path=adapter.SKILLS/'models/2015/ft_pct.json'
    if not model_path.exists():pytest.skip('Private fitted FT source unavailable')
    model=json.loads(model_path.read_text())
    end=pd.Timestamp('2016-03-01');cut=pd.Timestamp('2016-01-15')
    original=adapter.predict_ft_games(model,end)
    truncated=adapter.predict_ft_games(model,cut)
    keys=['game_id','player_id']
    a=original.loc[original.game_date.lt(cut)].set_index(keys).sort_index()
    b=truncated.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index,b.index)
    np.testing.assert_allclose(a.ft_expectation,b.ft_expectation,atol=1e-12,rtol=0)
    load=adapter.load_panel
    def mutated(*args):
        d,ctx=load(*args);mask=d.game_date.ge(cut)
        d.loc[mask,'ftm']=d.loc[mask,'fta']-d.loc[mask,'ftm']
        return d,ctx
    monkeypatch.setattr(adapter,'load_panel',mutated)
    changed=adapter.predict_ft_games(model,end).set_index(keys).sort_index()
    a=original.loc[original.game_date.le(cut)].set_index(keys).sort_index()
    np.testing.assert_allclose(a.ft_expectation,changed.reindex(a.index).ft_expectation,atol=1e-12,rtol=0)
    assert len(original.loc[original.game_date.eq(cut)])>0
    assert original.model_fit_end.lt(original.game_date).all()
    assert (original.last_shooter_observation.isna()|original.last_shooter_observation.lt(original.game_date)).all()


def test_saved_ft_seam_matches_expected_minus_actual_and_keeps_actual_files():
    path=adapter.OUT/'ft_2016_RS.parquet'
    if not path.exists():pytest.skip('Private FT seam unavailable')
    d=pd.read_parquet(path);actual=pd.read_parquet(adapter.ATOMS/'atoms_2016_RS.parquet')
    np.testing.assert_array_equal(pd.to_numeric(d.game_id),pd.to_numeric(actual.game_id))
    np.testing.assert_array_equal(d.event_num,actual.event_num)
    np.testing.assert_allclose(d.expected_ft_points,actual.actual_points+d.ft_delta,atol=1e-12)
    assert np.count_nonzero(d.ft_delta)>10000
    assert (d.expected_ft_points>=0).all()
