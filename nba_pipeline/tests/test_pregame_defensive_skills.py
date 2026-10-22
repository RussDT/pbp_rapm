"""Measurement semantics and chronology of selected defensive query recipes."""
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_pregame_defensive_skills import eb_queries, rim_queries, mpg_queries, reconcile, build_year
from build_predictive_rim_saved import lagged_league, rolling


def queries(days, players=None):
    return pd.DataFrame(dict(player_id=players or [1]*len(days),
        game_id=[str(i).zfill(10) for i in range(len(days))],
        game_date=pd.to_datetime(days, unit='D'), minutes=20., season_type='RS'))


def test_tracking_zero_observations_are_valid_missing_rows_are_queries():
    q=queries([1,2,3,4]); z=q.iloc[[0,2]][['player_id','game_id']].copy()
    z['numerator']=[0.,2.]; z['denominator']=[10.,10.]
    value,ex,last,valid=eb_queries(q,z,half_life=10)
    assert np.isnan(value[0]) and value[1]==0 and value[2]==0 and value[3]>0
    assert valid.tolist()==[True,False,True,False]
    assert ex[2]<ex[1] and last.tolist()==[-1,1,1,3]


def test_tracking_current_and_future_measurements_leave_prefix_identical():
    q=queries([1,2,2,3,4]); z=q[['player_id','game_id']].copy()
    z['numerator']=[1.,2.,3.,4.,5.];z['denominator']=10.
    a=eb_queries(q,z); z.loc[1:,'numerator']=0.; b=eb_queries(q,z)
    np.testing.assert_array_equal(a[0][:3],b[0][:3])
    assert a[0][1]==a[0][2] and a[0][3]!=b[0][3]
    assert a[2][3]==2


def test_tracking_query_only_dates_do_not_change_subsequent_states():
    q=queries([1,2,3,4]); z=q.iloc[[0,3]][['player_id','game_id']].copy()
    z['numerator']=[1.,2.];z['denominator']=10.
    a=eb_queries(q,z);b=eb_queries(q.iloc[[0,3]].reset_index(drop=True),z)
    np.testing.assert_allclose(a[0][[0,3]],b[0],equal_nan=True,atol=1e-13,rtol=0)


def test_rim_reproduces_selected_rolling_on_measured_dates():
    q=queries([1,5,9,11,15,22],players=[1,1,1,2,2,2])
    z=q[['player_id','game_id','game_date']].copy()
    z['rim_dfga']=[3.,4.,0.,6.,2.,1.];z['rim_dfgm']=[1.,3.,0.,4.,1.,0.];z['def_poss']=50.
    model=dict(volume_half=365,selected=dict(half_life=730,prior_attempts=100),mean=.06)
    a,last,_=rim_queries(q,z,model); z['league_pct']=lagged_league(z)
    b=rolling(z,365,730,100,.06)
    seen=last>=0
    np.testing.assert_allclose(a[seen,:3],b[seen,:3],rtol=0,atol=1e-14)
    assert np.isnan(a[~seen,:3]).all()


def test_rim_same_date_global_center_excludes_current_and_future():
    q=queries([1,2,2,3],players=[1,1,2,1]);z=q[['player_id','game_id']].copy()
    z['rim_dfga']=[2.,3.,4.,5.];z['rim_dfgm']=[1.,1.,2.,2.];z['def_poss']=50.
    model=dict(volume_half=365,selected=dict(half_life=730,prior_attempts=100),mean=.06)
    a=rim_queries(q,z,model)[0];z.loc[1:,'rim_dfgm']=0.;b=rim_queries(q,z,model)[0]
    np.testing.assert_array_equal(a[:3,:3],b[:3,:3]);assert a[3,2]!=b[3,2]


def test_mpg_uses_prior_365day_rs_positive_appearances_and_batches_dates():
    q=queries([1,2,2,3,366,367]);q['minutes']=[10.,20.,30.,0.,40.,50.]
    q.loc[1,'season_type']='PS'
    value,n,last=mpg_queries(q)
    assert np.isnan(value[0]);assert value[1]==value[2]==10.
    assert value[3]==value[4]==20.;assert value[5]==35.
    assert n.tolist()==[0,1,1,2,2,2] and last[3]==2
    changed=q.copy();changed.loc[1:,'minutes']=99.
    np.testing.assert_array_equal(mpg_queries(changed)[0][:3],value[:3])


def test_reconcile_retains_tracking_denominator_and_excludes_false_identity():
    q=queries([1]);q['def_poss']=49.
    z=pd.concat([q,q.assign(player_id=2)]).reset_index(drop=True)
    z['def_poss']=50.
    accepted,audit=reconcile(z,q,'def_poss')
    assert accepted.def_poss.tolist()==[50.]
    assert audit['missing_identity_rows']==1 and audit['differing_positive_denominators']==1
    z.loc[0,'game_date']=pd.Timestamp('2000-01-01')
    with pytest.raises(ValueError,match='date differs'):reconcile(z,q,'def_poss')


def test_arbitrary_cut_queries_exclude_same_date_and_include_new_players(tmp_path):
    q=queries([1,2]);q.game_date=pd.to_datetime(['2019-09-30','2019-10-01']);q['season']=2020
    rim=q[['player_id','game_id','game_date']].assign(rim_dfga=2.,rim_dfgm=1.,def_poss=50.)
    dfga=q[['player_id','game_id']].assign(dfga_value=10.,def_poss=50.)
    measured=pd.DataFrame(columns=['player_id','game_id','feature','numerator','denominator'])
    model=dict(vintage=2019,fit_cut='2019-10-01',
        rim=dict(fit_end='2019-06-01',volume_half=365,selected=dict(half_life=730,prior_attempts=100),mean=.06),
        dfga=dict(fit_end='2019-06-01',mean=.2,params=[100.,.01,10.,.002,.001,.25]))
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    a=build_year(tmp_path,2019,q,rim,dfga,measured,snapshot_cut='2019-10-01',snapshot_ids=[1,2],model_file=path)
    assert a.player_id.tolist()==[1,2]
    assert a.own_dfga100.notna().tolist()==[True,False]
    assert a.ctx_games.tolist()==[1,0]
    assert not a.own_rim_saved100__observed_this_game.any()
    assert a.own_dfga100__prior_exposure.tolist()==[50.,0.]
    assert a.own_dfga100__last_observation_day.iloc[0] < (pd.Timestamp('2019-10-01')-pd.Timestamp('1970-01-01')).days
