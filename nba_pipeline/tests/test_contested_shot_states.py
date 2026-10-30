from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_contested_shot_states as C
import build_vpm_defense_bank as B
import build_full_skill_history as F


def source():
    return pd.DataFrame({'player_id':[1,2,1,2],'game_id':['1','1','2','2'],'game_date':pd.to_datetime(['2016-10-25','2016-10-25','2016-10-26','2016-10-26']),'season':[2017]*4,'def_poss':[40.,30.,50.,20.],C.RAW[0]:[5.,4.,0.,3.],C.RAW[1]:[3.,3.,0.,1.],C.RAW[2]:[2.,1.,0.,2.]})


def queries(days=('2016-10-25','2016-10-26','2016-10-27'),ids=(1,2,999)):
    return pd.DataFrame([dict(query_id=f'{date}:{pid}',player_id=pid,query_date=date,query_season=2017,cohort_vintage=2016) for date in days for pid in ids])


def modern():
    p=source();m=p[['player_id','game_id','game_date','def_poss',C.RAW[0]]].rename(columns={'def_poss':'denominator',C.RAW[0]:'numerator'});m['feature']=C.MODERN[0]
    a=p[['player_id','game_date',*C.RAW]].rename(columns=dict(zip(C.RAW,C.ARCHIVE)))
    return m,a


def test_exact_count_closure_and_observed_zero_denominator_missingness():
    p=source();r=C.audit_observations(p);assert r['observed_triplets']==4 and r['zeros'][C.HEADS[0]]==1 and r['maximum_raw_split_closure_gap']==0
    p.loc[0,'def_poss']=np.nan;r=C.audit_observations(p);assert r['observed_but_unusable_denominator_rows']==1 and r['admissible_positive_denominator_rows']==3
    for field,value in [(C.RAW[0],6),(C.RAW[2],np.nan),(C.RAW[1],-.5),(C.RAW[1],np.inf)]:
        bad=source();bad.loc[0,field]=value
        with pytest.raises(ValueError):C.audit_observations(bad)
    for field,value in [('season',2027),('game_date',pd.Timestamp('2026-10-01')),('player_id',-1)]:
        bad=source();bad.loc[0,field]=value
        with pytest.raises(ValueError):C.audit_observations(bad)


def test_modern_source_reconciliation_retains_original_denominator_and_rejects_guesses():
    m,a=modern();m.loc[0,'denominator']=39.5;wide,audit=C.modern_split_measurements(m,a)
    assert wide.denominator.iloc[0]==39.5 and audit['missing_or_mismatched_original_total_rows']==0
    np.testing.assert_array_equal(wide[C.MODERN[0]],wide[C.MODERN[1]]+wide[C.MODERN[2]])
    for bad in [a.iloc[1:],pd.concat([a,a.iloc[:1]]),a.assign(hustle_CONTESTED_SHOTS_2PT=99)]:
        with pytest.raises(ValueError):C.modern_split_measurements(m,bad)


def test_modern_same_estimator_total_matches_literal_frozen_summarize_and_splits_close():
    m,a=modern();wide,_=C.modern_split_measurements(m,a);q=queries();out=C.modern_query_states(wide,q)
    assert out.loc[out.query_date.eq('2016-10-25'),C.MODERN].isna().all().all()
    for day,g in out.groupby('query_date'):
        old,_=B.summarize(m,g.player_id,day,half_life=550.,prior_exposure=1000.)
        np.testing.assert_allclose(g[C.MODERN[0]],old[C.MODERN[0]],rtol=0,atol=1e-12,equal_nan=True)
    seen=out.player_id.ne(999)&out.query_date.gt('2016-10-25');np.testing.assert_allclose(out.loc[seen,C.MODERN[0]],out.loc[seen,C.MODERN[1]]+out.loc[seen,C.MODERN[2]],atol=1e-12,rtol=0)
    assert out.loc[out.player_id.eq(999),C.MODERN].isna().all().all()
    zero=out.loc[out.player_id.eq(1)&out.query_date.eq('2016-10-27')].iloc[0]
    assert zero[C.MODERN[0]+'__last_observation_date']==pd.Timestamp('2016-10-26') and zero[C.MODERN[0]+'__exposure']>89


def test_modern_same_day_future_mutations_and_added_queries_do_not_change_prefix():
    m,a=modern();w,_=C.modern_split_measurements(m,a);q=queries(('2016-10-26',));before=C.modern_query_states(w,q)
    changed=w.copy();changed.loc[changed.game_date.ge('2016-10-26'),C.MODERN]=[900.,600.,300.]
    pd.testing.assert_frame_equal(before,C.modern_query_states(changed,q),check_exact=True)
    full=C.modern_query_states(w,queries());pd.testing.assert_frame_equal(before.reset_index(drop=True),full.loc[full.query_date.eq('2016-10-26')].reset_index(drop=True),check_exact=True)
    with pytest.raises(ValueError):C.modern_query_states(w,q.assign(query_season=2027))


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('contested-kernels');return C.D.compile_kernel(tmp/'query'),F.compile_kernel(tmp/'original')


def recipes():
    return {2016:{h:dict(year=2016,training_end_season=2015,status='bootstrap',last_training_observation=None,parameters=[1500.,.01,30.,.002,.002,1.+j],mean=.05+j*.01) for j,h in enumerate(C.HEADS)}}


def terminal(panel,q,spec,recipe,kernel):
    cut=pd.Timestamp(q.query_date);p=C.normalize_source(panel);p=p.loc[p.game_date.lt(cut)].copy();p['_q']=False
    row={c:np.nan for c in p};row.update(player_id=q.player_id,season=q.query_season,game_date=cut,day=int((cut-pd.Timestamp('1970-01-01')).days),_q=True)
    full=pd.concat([p,pd.DataFrame([row])],ignore_index=True).sort_values(['player_id','game_date','_q'],kind='stable');engine=F.Engine(full,spec,kernel);values=engine.run(recipe['parameters'],recipe['mean']);ix=np.flatnonzero(full._q)[0]
    return values[0][ix]*100 if values[3][ix]>0 else np.nan,values[3][ix],values[5][ix]


def test_OU_transport_matches_literal_terminal_kernel_and_keeps_independent_total(kernels):
    query,original=kernels;q=queries();out,prov=C.ou_query_states(source(),q,recipes(),query,original)
    for row in C.D.normalize_queries(q).itertuples():
        j=out.query_id.eq(row.query_id)
        for spec in C.SPECS:
            value,e,last=terminal(source(),row,spec,recipes()[2016][spec.name],original)
            np.testing.assert_allclose(out.loc[j,'lib__'+spec.name],value,atol=1e-12,rtol=0,equal_nan=True)
            np.testing.assert_allclose(out.loc[j,'lib__'+spec.name+'__exposure'],e,atol=0,rtol=0)
    assert out.loc[out.player_id.eq(999),['lib__'+h for h in C.HEADS]].isna().all().all()
    assert out.diag__independent_total_minus_sum_parts100.abs().max()>1e-5
    assert prov.loc[prov.player_observed,'last_observation_date'].lt(prov.loc[prov.player_observed,'query_date']).all()


def test_OU_same_day_future_perturbation_zero_history_and_empty_queries(kernels):
    query,original=kernels;q=queries(('2016-10-26',));before,_=C.ou_query_states(source(),q,recipes(),query,original)
    bad=source();bad.loc[bad.game_date.ge('2016-10-26'),C.RAW]=[900.,600.,300.]
    after,_=C.ou_query_states(bad,q,recipes(),query,original);pd.testing.assert_frame_equal(before,after,check_exact=True)
    empty,provenance=C.ou_query_states(source(),q.iloc[:0],recipes(),query,original);assert len(empty)==len(provenance)==0
    wrong=recipes();wrong[2016][C.HEADS[0]]['last_training_observation']='2016-10-26'
    with pytest.raises(ValueError,match='training observation'):C.ou_query_states(source(),q,wrong,query,original)


def test_annual_packet_values_zero_and_metadata_preserved_without_total_reconciliation():
    q=queries(('2017-10-01',),(1,));q['query_season']=2017;q['cohort_vintage']=2017;p=q.copy();p['query_date']=pd.to_datetime(p.query_date)
    for j,h in enumerate(C.HEADS):
        p['lib__'+h]=float(j);p['lib__'+h+'__exposure']=100.;p['lib__'+h+'__last_observation_date']=pd.Timestamp('2017-05-01')
    out=C.annual_packet_features(p,q);assert out['lib__'+C.HEADS[0]].iloc[0]==0 and not out['lib__'+C.HEADS[0]+'__missing'].iloc[0]
    assert out.diag__independent_total_minus_sum_parts100.iloc[0]==-3
    bad=p.copy();bad.loc[0,'lib__'+C.HEADS[0]+'__last_observation_date']=pd.Timestamp('2017-10-01')
    with pytest.raises(ValueError):C.annual_packet_features(bad,q)
    with pytest.raises(ValueError,match='Protected'):C.annual_packet_features(pd.concat([p,p.assign(query_season=2027)]),q)


def test_missing_or_infinite_annual_exposure_and_noncanonical_modern_source_fail_closed():
    q=queries(('2017-10-01',),(1,));q['query_season']=2017;p=q.copy()
    for h in C.HEADS:
        p['lib__'+h]=0.;p['lib__'+h+'__exposure']=20.;p['lib__'+h+'__last_observation_date']=pd.Timestamp('2017-06-01')
    for value in [np.nan,np.inf,0.,-1.]:
        bad=p.copy();bad['lib__'+C.HEADS[0]+'__exposure']=value
        with pytest.raises(ValueError,match='physical history'):C.annual_packet_features(bad,q)
    m,a=modern();wide,_=C.modern_split_measurements(m,a)
    for value in [1.5,-1.,np.nan]:
        bad=wide.copy();bad['player_id']=bad.player_id.astype(float);bad.loc[0,'player_id']=value
        with pytest.raises(ValueError,match='source identity'):C.modern_query_states(bad,queries())
