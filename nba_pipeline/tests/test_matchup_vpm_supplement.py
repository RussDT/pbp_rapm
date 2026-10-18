import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('supplement',Path(__file__).parents[1]/'scripts/audit_matchup_vpm_supplement.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def packet(year,q=None,actual=None):
    q=np.array([[1.],[2.]]) if q is None else np.asarray(q,float)
    actual=np.array([3.,5.]) if actual is None else np.asarray(actual,float)
    return dict(season=year,cutoff=f'{year-1}-10-01',last_game=f'{year}-06-15',
                design=q,actual=actual,base=np.zeros(len(actual)))


def test_null_first_removes_warmup_and_has_exact_zero_first_development_increment():
    packets=[packet(2019),packet(2020),packet(2021)]
    assert m.training_prefix(packets,packets[1])==[]
    assert [p['season'] for p in m.training_prefix(packets,packets[2])]==[2020]
    params=dict(ridge=1.,strength=1.,group_weights=[1.])
    np.testing.assert_array_equal(m.qr_coefficients([],params,[0]),[0.])
    packets[0]['actual'][:]=1e12
    assert [p['season'] for p in m.training_prefix(packets,packets[2])]==[2020]


def test_training_prefix_rejects_unfinished_labels_and_future_labels_cannot_affect_qr():
    packets=[packet(2020),packet(2021),packet(2022)]
    params=dict(ridge=1.,strength=1.,group_weights=[1.])
    before=m.qr_coefficients(m.training_prefix(packets,packets[1]),params,[0])
    packets[1]['actual'][:]=1e12;packets[2]['actual'][:]=-1e12
    np.testing.assert_array_equal(before,m.qr_coefficients(m.training_prefix(packets,packets[1]),params,[0]))
    packets[0]['last_game']=packets[1]['cutoff']
    with pytest.raises(ValueError,match='Unfinished'):
        m.training_prefix(packets,packets[1])


def test_augmented_qr_matches_scalar_equal_season_closed_form_with_group_scaling():
    history=[packet(2020),packet(2021,[[4.]],[7.])]
    params=dict(ridge=.7,strength=.4,group_weights=[.3])
    # One season has2games, the other1: each season contributes half the risk.
    rhs=(np.mean(np.array([1.,2.])*np.array([3.,5.]))+4*7)/2
    gram=(np.mean(np.array([1.,2.])**2)+4**2)/2
    expected=.4*.3*(.3*rhs)/(.3**2*gram+.7)
    assert m.qr_coefficients(history,params,[0])[0]==pytest.approx(expected)
    assert m.qr_coefficients(history,{**params,'strength':0},[0])[0]==0


def test_actual_points_use_native_common_lineups_positive_good_defense_and_raw_intercept():
    row={column:i+1 for i,column in enumerate(m.SLOTS)}
    atoms=pd.DataFrame([{**row,'game_id':'g1','actual_points':2.},
                        {**row,'game_id':'g1','actual_points':3.},
                        {**row,'O1':11,'game_id':'g2','actual_points':2.}])
    base=pd.DataFrame({'player_id':range(1,11),'raw_ovpm':100.,'raw_dvpm':20.}).set_index('player_id')
    improved=base.copy();improved.raw_ovpm=80.
    got,n=m.point_errors(atoms,{m.BASE:base,'candidate':improved},1.)
    assert n==2 and len(got)==1 and got.rows.iloc[0]==2
    # Baseline predicts1+5*1-5*.2=5; candidate predicts4.
    assert got[m.BASE+'_squared_error'].iloc[0]==13
    assert got.candidate_squared_error.iloc[0]==5
    bad=base.drop(index=1)
    with pytest.raises(ValueError,match='common-native'):
        m.point_errors(atoms,{m.BASE:base,'candidate':bad},1.)
    atoms.loc[0,'actual_points']=2.5
    with pytest.raises(ValueError,match='integer'):
        m.point_errors(atoms,{m.BASE:base},1.)


def test_paired_bootstrap_equal_season_ratio_and_gain_sign_on_unequal_game_exposure():
    rows=[]
    for year in range(2020,2027):
        gain=float(year-2019)
        for game,n in enumerate([1,9]):
            row=dict(season=year,game_id=f'{year}_{game}',rows=n)
            row[m.BASE+'_squared_error']=20*n
            for index,name in enumerate(m.ARMS.values()):
                row[name+'_squared_error']=(20-gain-index)*n
            rows.append(row)
    result=m.bootstrap_reference(pd.DataFrame(rows),seed=91,reps=30)
    assert len(result)==8
    for row in result.itertuples():
        shift=list(m.ARMS.values()).index(row.primary)
        expected=(2. if row.period=='development_2020_2022' else 5.5)+shift
        assert row.mse_gain==pytest.approx(expected)
        assert row.mse_gain_lo==pytest.approx(expected)
        assert row.mse_gain_hi==pytest.approx(expected)
        assert row.reference_mse==20


def test_saved_outputs_are_required_before_any_supplemental_scoring(tmp_path):
    with pytest.raises(ValueError,match='Wait for pipeline-generated'):
        m.audit(tmp_path)
    assert not list(tmp_path.iterdir())


def test_nonconstant_cluster_bootstrap_uses_frozen_seed_and_keeps_arms_paired():
    records=[]
    for year in range(2020,2027):
        for game,count in enumerate([1,2,8]):
            record=dict(season=year,game_id=f'{year}_{game}',rows=count)
            record[m.BASE+'_squared_error']=10*count
            for name in m.ARMS.values():
                record[name+'_squared_error']=10*count-[0,3,1][game]
            records.append(record)
    errors=pd.DataFrame(records)
    a=m.bootstrap_reference(errors,17,31)
    b=m.bootstrap_reference(errors,17,31)
    c=m.bootstrap_reference(errors,18,31)
    pd.testing.assert_frame_equal(a,b)
    assert (a.mse_gain_hi>a.mse_gain_lo).all()
    assert not np.array_equal(a.mse_gain_lo.to_numpy(),c.mse_gain_lo.to_numpy())
    # Point estimate is ratio of paired SSE sums to possession count in each
    # season: 4/11. It is not the average per-game rate.
    np.testing.assert_allclose(a.mse_gain,4/11,rtol=0,atol=1e-12)


def test_compare_rejects_missingness_key_and_value_changes():
    a=pd.DataFrame({'game_id':['g'],'loss':[1.]})
    with pytest.raises(ValueError,match='census'):
        m.compare_frame(a,pd.DataFrame({'game_id':['x'],'loss':[1.]}),['game_id'],['loss'],'test')
    with pytest.raises(ValueError,match='mismatch'):
        m.check([np.nan],[0.],'missingness')
