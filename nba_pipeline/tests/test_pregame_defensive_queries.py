"""Explicit NBA season and batched pure-query semantics for defensive inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_pregame_defensive_queries as Q
import build_pregame_defensive_skills as F


@pytest.fixture
def inputs(tmp_path):
    dates=pd.to_datetime(['2019-09-20','2019-10-02','2020-09-30','2020-10-05','2021-01-01','2026-06-01'])
    q=pd.DataFrame(dict(player_id=1,game_id=[str(i).zfill(10) for i in range(len(dates))],
        game_date=dates,season=[2019,2020,2020,2020,2021,2026],season_type='RS',minutes=[10.,20.,30.,40.,50.,60.]))
    rim=q[F.KEY+['game_date']].assign(rim_dfga=2.,rim_dfgm=1.,def_poss=50.)
    dfga=q[F.KEY].assign(dfga_value=10.,def_poss=50.)
    measurements=pd.concat([q[F.KEY].assign(feature=name,numerator=1.,denominator=50.) for name in F.TRACKING],ignore_index=True)
    for year in [2019,2025]:
        model=dict(vintage=year,fit_cut=f'{year}-10-01',
            rim=dict(fit_end=f'{year}-06-01',volume_half=365,selected=dict(half_life=730,prior_attempts=100),mean=.06),
            dfga=dict(fit_end=f'{year}-06-01',mean=.2,params=[100.,.01,10.,.002,.001,.25]))
        (tmp_path/f'{year}.json').write_text(json.dumps(model))
    return dict(queries=q,rim=rim,dfga=dfga,measurements=measurements,models_root=tmp_path)


def query(dates,seasons,players=None):
    return pd.DataFrame(dict(player_id=players or [1]*len(dates),query_date=pd.to_datetime(dates),
        query_season=seasons,query_id=[str(i) for i in range(len(dates))]))


def run(requested,inputs,tmp_path,year=2019):
    return Q.query_vintage(requested,year,inputs,tmp_path/'kernel')[0]


def test_explicit_bubble_season_controls_query_variance_without_calendar_inference(inputs,tmp_path):
    right=run(query(['2020-10-01'],[2020]),inputs,tmp_path)
    alternate=run(query(['2020-10-01'],[2021]),inputs,tmp_path)
    assert right.query_season.tolist()==[2020]
    assert right.parameter_vintage.tolist()==[2019]
    assert alternate.own_dfga100__state_variance.iloc[0]>right.own_dfga100__state_variance.iloc[0]


def test_query_only_season_and_time_never_commit_to_later_observation(inputs,tmp_path):
    sparse=run(query(['2020-10-05'],[2020]),inputs,tmp_path)
    full=run(query(['2020-10-01','2020-10-02','2020-10-05'],[2021,2021,2020]),inputs,tmp_path)
    for name in Q.FEATURES+['own_dfga100__state_variance']:
        np.testing.assert_allclose(full[name].iloc[-1],sparse[name].iloc[0],atol=1e-12,rtol=0,equal_nan=True)
    assert full.own_dfga100__last_observation_day.iloc[-1]==(pd.Timestamp('2020-09-30')-pd.Timestamp('1970-01-01')).days


def test_completed_september_cut_retains_explicit_season2026(inputs,tmp_path):
    result=run(query(['2026-09-01'],[2026]),inputs,tmp_path,year=2025)
    assert result.query_season.tolist()==[2026]
    assert result.parameter_vintage.tolist()==[2025]
    with pytest.raises(ValueError,match='out-of-scope'):
        run(query(['2026-09-01'],[2027]),inputs,tmp_path,year=2025)


def test_actual_game_query_matches_frozen_export_values_and_missingness(inputs,tmp_path):
    out=tmp_path/'reference';out.mkdir();(out/'run_lock.json').write_text('{}')
    F.build_year(out,2019,inputs['queries'],inputs['rim'],inputs['dfga'],inputs['measurements'],
        model_file=inputs['models_root']/'2019.json')
    reference=pd.read_parquet(out/'vintages/2019.parquet')
    requested=reference.rename(columns={'game_date':'query_date','season':'query_season'})[
        ['player_id','query_date','query_season']]
    result=run(requested,inputs,tmp_path)
    for name in Q.FEATURES:
        for suffix in ['', '_missing','__prior_exposure','__last_observation_day']:
            np.testing.assert_allclose(result[name+suffix],reference[name+suffix],atol=1e-12,rtol=0,equal_nan=True)
    assert not result.filter(like='__observed_this_game').to_numpy().any()


def test_annual_snapshot_matches_frozen_helper_when_season_metadata_agrees(inputs,tmp_path):
    old=F.build_year(tmp_path,2019,inputs['queries'],inputs['rim'],inputs['dfga'],inputs['measurements'],
        snapshot_cut='2019-10-01',snapshot_ids=[1,2],model_file=inputs['models_root']/'2019.json')
    new=run(query(['2019-10-01']*2,[2020]*2,[1,2]),inputs,tmp_path)
    for name in Q.FEATURES:
        np.testing.assert_allclose(new[name],old[name],atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(new[name+'_missing'],old[name+'_missing'])


def test_complete_current_future_measurement_perturbation_leaves_prefix_exact(inputs,tmp_path):
    requested=query(['2019-10-01','2019-10-02','2019-10-03'],[2020]*3)
    a=run(requested,inputs,tmp_path)
    changed={k:(v.copy() if isinstance(v,pd.DataFrame) else v) for k,v in inputs.items()}
    cutoff=pd.Timestamp('2019-10-02')
    changed['queries'].loc[changed['queries'].game_date.ge(cutoff),'minutes']=1.
    changed['rim'].loc[changed['rim'].game_date.ge(cutoff),'rim_dfgm']=0.
    future_ids=set(changed['queries'].loc[changed['queries'].game_date.ge(cutoff),'game_id'])
    changed['dfga'].loc[changed['dfga'].game_id.isin(future_ids),'dfga_value']=0.
    changed['measurements'].loc[changed['measurements'].game_id.isin(future_ids),'numerator']=0.
    b=run(requested,changed,tmp_path)
    pd.testing.assert_frame_equal(a.iloc[:2],b.iloc[:2],check_exact=True)
    assert a.own_dfga100.iloc[-1]!=b.own_dfga100.iloc[-1]
    assert a.own_rim_saved100.iloc[-1]!=b.own_rim_saved100.iloc[-1]
    assert a.ctx_mpg.iloc[-1]!=b.ctx_mpg.iloc[-1]


def test_same_day_queries_share_prior_and_unseen_players_stay_missing(inputs,tmp_path):
    result=run(query(['2019-10-02']*3,[2020]*3,[1,1,2]),inputs,tmp_path)
    for name in Q.FEATURES:
        assert result[name].iloc[0]==result[name].iloc[1]
        assert np.isnan(result[name].iloc[2])
        assert result[name+'__prior_exposure'].iloc[2]==0
        assert result[name+'__last_observation_day'].iloc[2]==-1


def test_actual_same_date_season_conflict_and_duplicate_query_identity_fail(inputs,tmp_path):
    with pytest.raises(ValueError,match='conflicts with actual'):
        run(query(['2020-10-05'],[2021]),inputs,tmp_path)
    requested=query(['2019-10-01','2019-10-02'],[2020,2020]);requested['query_id']='same'
    with pytest.raises(ValueError,match='unique'):run(requested,inputs,tmp_path)
