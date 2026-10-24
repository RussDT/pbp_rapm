from pathlib import Path
import subprocess,sys
import numpy as np
import pandas as pd
import pytest
S=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(S))
import export_vpm_1997_daily_features as D
import export_vpm_1997_features as A
from build_full_skill_history import Engine
from full_skill_history_specs import Skill,SKILLS

@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    out=tmp_path_factory.mktemp('dailyquery');old=out/'old.so'
    subprocess.run(['c++','-O2','-std=c++11','-shared','-fPIC',str(S/'full_skill_history_kernel.cpp'),'-o',str(old)],check=True)
    return old,D.compile_kernel(out)

def q(dates=('1997-01-01',),ids=None,seasons=None):
    ids=ids or [1]*len(dates);seasons=seasons or [pd.Timestamp(x).year for x in dates]
    return D.normalize_queries(pd.DataFrame([dict(query_id=f'q:{i}:{p}:{d}',player_id=p,query_date=d,query_season=s,cohort_vintage=s) for i,(p,d,s) in enumerate(zip(ids,dates,seasons))]))

def panel():
    p=pd.DataFrame(dict(player_id=[1]*6,game_id=list('abcdef'),game_date=pd.to_datetime(['1996-11-01','1997-01-01','1997-01-01','1997-06-01','1997-11-02','1998-02-01']),season=[1997]*4+[1998]*2,**{'core.ftm':[0.,1.,0.,np.nan,2.,3.],'core.fta':[4.,2.,0.,np.nan,4.,4.]}))
    p['day']=(p.game_date-pd.Timestamp('1970-01-01')).dt.days
    return p

def ft():return Skill('ft_pct','core.ftm','core.fta','shooting','pct',.75)
def recipe():return dict(year=1997,training_end_season=1996,status='bootstrap',mean=.75,parameters=[1500.,.03,30.,.002,.4,.2])

def test_calendar_native_season_is_separate_and_held_report_vintage():
    assert D.parameter_vintage('1996-11-01')==1997
    assert D.parameter_vintage('1997-10-01')==1997
    assert D.parameter_vintage('1998-02-01')==1997
    assert D.parameter_vintage('2020-10-15')==2019
    assert D.parameter_vintage('2020-11-01')==2020
    assert D.parameter_vintage('2026-10-01')==2026
    assert q(['1997-11-02'],seasons=[1998]).parameter_vintage.iloc[0]==1997
    with pytest.raises(ValueError):D.normalize_queries(q().assign(parameter_vintage=1998))
    for date in ['1996-10-31','2026-10-02','2027-01-01']:
        with pytest.raises(ValueError):q([date])
    with pytest.raises(ValueError):q(['2026-06-01'],seasons=[2027])
    with pytest.raises(ValueError):D.normalize_queries(pd.concat([q(),q()]))


def test_pure_daily_matches_independent_terminal_prefixes_exactly(kernels):
    old,new=kernels;p=panel();queries=q(['1996-11-01','1997-01-01','1997-03-01','1997-07-01','1997-10-01','1997-11-02','1998-03-01'],seasons=[1997]*5+[1998]*2)
    result=D.QueryEngine(p,queries,new).run(ft(),recipe(),old)
    for j,row in queries.iterrows():
        prefix=p.loc[p.game_date.lt(row.query_date)].copy();terminal=pd.DataFrame([dict(player_id=row.player_id,game_date=row.query_date,season=row.query_season,day=(row.query_date-pd.Timestamp('1970-01-01')).days,**{'core.ftm':np.nan,'core.fta':np.nan})])
        full=pd.concat([prefix,terminal],ignore_index=True);e=Engine(full,ft(),old);r=e.run(recipe()['parameters'],recipe()['mean'])
        assert result['prior_exposure'][j]==r[3][-1] and result['last_day'][j]==r[5][-1]
        expected=r[0][-1] if r[3][-1]>0 else np.nan
        np.testing.assert_equal(result['prediction'][j],expected)
    assert result['prior_exposure'][1]==4 # current same-date batch excluded
    assert result['prior_exposure'][2]==6 # zero-opportunity source row adds nothing


def test_query_density_order_cold_player_future_and_same_day_invariance(kernels):
    old,new=kernels;p=panel();dates=['1997-01-01','1997-11-02','1998-03-01'];queries=q(dates,seasons=[1997,1998,1998]);a=D.QueryEngine(p,queries,new).run(ft(),recipe(),old)
    extra=q(['1996-11-01','1997-07-01','1997-08-01','1998-02-01'],ids=[999,1,1,999],seasons=[1997,1997,1998,1998]);extra.query_id='extra:'+extra.query_id
    full=pd.concat([extra,queries],ignore_index=True).sample(frac=1,random_state=3).reset_index(drop=True);b=D.QueryEngine(p,full,new).run(ft(),recipe(),old)
    ix=full.query_id.get_indexer(queries.query_id) if hasattr(full.query_id,'get_indexer') else pd.Index(full.query_id).get_indexer(queries.query_id)
    for key in a:np.testing.assert_equal(a[key],b[key][ix])
    changed=p.copy();changed.loc[changed.game_date.ge('1997-11-02'),['core.ftm','core.fta']]=999
    early=q(['1997-11-02'],seasons=[1998]);c=D.QueryEngine(changed,early,new).run(ft(),recipe(),old)
    assert c['prediction'][0]==a['prediction'][1]
    cold=D.QueryEngine(p,q(['1996-11-01'],ids=[999],seasons=[1997]),new).run(ft(),recipe(),old)
    assert np.isnan(cold['prediction'][0]) and cold['prior_exposure'][0]==0


def test_real_missing_row_is_not_treated_as_synthetic_query(kernels):
    old,new=kernels;p=panel();queries=q(['1998-03-01'],seasons=[1998]);a=D.QueryEngine(p,queries,new).run(ft(),recipe(),old)
    # Missing source dates are genuine events in the frozen estimator, unlike
    # injected query dates. A season boundary there changes next update gain.
    p.loc[3,'season']=1998
    b=D.QueryEngine(p,queries,new).run(ft(),recipe(),old)
    assert a['prediction'][0]!=b['prediction'][0]


def test_source_epoch_and_missing_v2_never_fills_v1(kernels):
    old,new=kernels;p=panel();s=Skill('later','core.ftm','core.fta','tracking','pct',.5,first_season=2014)
    r=D.QueryEngine(p,q(),new).run(s,recipe(),old)
    assert not r['source_supported'][0] and np.isnan(r['prediction'][0])
    # Support uses NBA query season, never the held parameter vintage.
    p.game_date=pd.to_datetime(['2023-11-01','2024-01-01','2024-01-01','2024-06-01','2024-11-02','2025-02-01']);p.season=[2024]*4+[2025]*2;p.day=(p.game_date-pd.Timestamp('1970-01-01')).dt.days
    s=Skill('tracking','core.ftm','core.fta','tracking','pct',.5,last_season=2024)
    r=D.QueryEngine(p,q(['2024-11-02'],seasons=[2025]),new).run(s,recipe(),old)
    assert not r['source_supported'][0] and r['player_observed'][0] and np.isnan(r['prediction'][0])


def test_compact_formulas_and_support_match_frozen_annual_exactly():
    date='1997-10-01';queries=q([date]);aq=queries[A.KEYS]
    values=pd.DataFrame({s.name:[.5 if s.kind=='pct' else 1.] for s in SKILLS});rows=[];heads={}
    for s in SKILLS:
        r=aq.iloc[0].to_dict();r.update(head_name=s.name,prior_exposure=10.,last_observation_date=pd.Timestamp('1997-09-01'),source_supported=True,player_observed=True);rows.append(r)
        if s.name in D.HEAD_NAMES:heads[s.name]=dict(prediction=values[s.name].to_numpy(float),prior_exposure=np.array([10.]),last_day=np.array([(pd.Timestamp('1997-09-01')-pd.Timestamp('1970-01-01')).days]),source_supported=np.array([True]),player_observed=np.array([True]))
    support=pd.DataFrame(dict(player_id=[1,1,1],game_date=pd.to_datetime(['1996-10-01','1996-10-02','1997-09-01']),playoffs=False,off_poss=[3.,4.,5.],def_poss=[4.,5.,6.],minutes=[20.1,21.2,22.3],minute_observed=True))
    wide=A.make_features(values,pd.DataFrame(rows),aq,A.feature_schema());expected=A.compact_features(wide,A.feature_schema(),support)
    now=D.compact_from_heads(queries,heads,D.SupportIndex(support))
    pd.testing.assert_frame_equal(now[expected.columns],expected,check_exact=True)
    heads['oreb100']['prediction'][0]=0;now=D.compact_from_heads(queries,heads,D.SupportIndex(support));assert now.hist__oreb100.iloc[0]==0 and not now.hist__oreb100__missing.iloc[0]
    heads['dreb_contest100__v2']['prediction'][0]=np.nan;heads['dreb_contest100__v2']['source_supported'][0]=False
    modern=q(['2025-10-01'],seasons=[2025]);now=D.compact_from_heads(modern,heads,D.SupportIndex(support));assert np.isnan(now.hist__contested_dreb100.iloc[0])


def test_mpg_zero_opportunity_unknown_history_and_current_day_gates():
    support=pd.DataFrame(dict(player_id=[1]*5,game_date=pd.to_datetime(['1996-01-01','1996-11-01','1997-05-01','1997-05-01','1997-11-01']),playoffs=[False,False,True,False,False],off_poss=[99.,0.,np.nan,4.,100.],def_poss=[99.,0.,np.nan,5.,100.],minutes=[99.,0.,np.nan,30.,100.],minute_observed=[True,True,False,True,True]))
    queries=q(['1997-11-01','1997-11-01'],ids=[1,999],seasons=[1998,1998]);m=D.SupportIndex(support).query(queries)
    assert m.archive_off_possessions_lower_bound.tolist()==[103.,0.] and m.archive_exposure_unknown_rows.tolist()==[1,0]
    assert m.archive_observed_games.tolist()==[4,0] and m.ctx_games.tolist()==[1,0]
    assert m.ctx_mpg.iloc[0]==30 and np.isnan(m.ctx_mpg.iloc[1])
    assert m.last_archive_observation_date.iloc[0]==pd.Timestamp('1997-05-01')


def test_query_census_has_no_future_identity_selection_and_covers_calendar():
    p=panel();p.loc[5,'player_id']=999
    c=D.build_calendar(p,include_reports=False);x=D.build_queries(p,c)
    assert x.query_date.min()==pd.Timestamp('1997-01-01')
    assert not x.player_id.eq(999).any() # unseen at final source date
    first=D.build_queries(p,c.iloc[:1]);assert len(first)==0
    bad=pd.DataFrame(dict(game_date=[pd.Timestamp('1997-01-01')],season=[2027]))
    with pytest.raises(ValueError):D.build_calendar(p,bad)


def test_store_rejects_actual_fit_end_reaching_query_before_replay(kernels):
    old,new=kernels;store=object.__new__(D.DailyFeatureStore)
    store.panel=panel();store.library=new;store.original_library=old
    bad=recipe();bad.update(status='annual_past_only_fit',last_training_observation='1997-01-01')
    store.recipes={1997:{s.name:bad for s in D.SPECS}}
    with pytest.raises(ValueError,match='training observation'):
        store.query_heads(q())


def test_export_preserves_disk_reserve_without_creating_output(tmp_path,monkeypatch):
    store=object.__new__(D.DailyFeatureStore)
    monkeypatch.setattr(D.shutil,'disk_usage',lambda p:type('Usage',(),{'free':14*1024**3})())
    dest=tmp_path/'blocked'
    with pytest.raises(OSError,match='reserve'):store.export(q(),dest)
    assert not dest.exists()


def fake_store():
    store=object.__new__(D.DailyFeatureStore);store.inputs={};store.registry=[];store.masks={};store.minute_audit={}
    store.verify_inputs=lambda:None;store.query=lambda frame:frame.copy()
    return store


def test_guarded_resume_never_recomputes_and_rejects_code_query_batch_tamper(tmp_path):
    store=fake_store();queries=q(['1997-01-01','1997-02-01']);root=tmp_path/'export'
    first=store.export(queries,root,batch_rows=1,minimum_free_bytes=0)
    store.query=lambda frame:(_ for _ in ()).throw(AssertionError('must not replay completed partition'))
    again=store.export(queries,root,batch_rows=1,minimum_free_bytes=0,resume=True);assert again==first
    with pytest.raises(ValueError,match='batch_rows'):store.export(queries,root,batch_rows=2,minimum_free_bytes=0,resume=True)
    changed=queries.copy();changed.loc[0,'player_id']=99
    with pytest.raises(AssertionError):store.export(changed,root,batch_rows=1,minimum_free_bytes=0,resume=True)
    store.inputs={'/changed/code.py':'bad'}
    with pytest.raises(ValueError,match='input_hashes'):store.export(queries,root,batch_rows=1,minimum_free_bytes=0,resume=True)


def test_interrupted_partition_keeps_completed_part_and_partial_evidence(tmp_path):
    store=fake_store();queries=q(['1997-01-01','1997-02-01']);root=tmp_path/'partial';calls=[]
    def query(frame):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('fixture interrupt before second writer')
        return frame.copy()
    store.query=query
    with pytest.raises(RuntimeError):store.export(queries,root,batch_rows=1,minimum_free_bytes=0)
    first=root/'partitions/1997/0000.parquet';h=D.sha(first)
    store.query=lambda frame:frame.copy();r=store.export(queries,root,batch_rows=1,minimum_free_bytes=0,resume=True)
    assert D.sha(first)==h and len(r['partitions'])==2
    broken=tmp_path/'broken';s=fake_store();calls=[]
    def query2(frame):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('interrupt')
        return frame.copy()
    s.query=query2
    with pytest.raises(RuntimeError):s.export(queries,broken,batch_rows=1,minimum_free_bytes=0)
    p=broken/'partitions/1997/0001.parquet.partial';p.write_bytes(b'preserved partial bytes')
    with pytest.raises(ValueError,match='Incomplete atomic'):s.export(queries,broken,batch_rows=1,minimum_free_bytes=0,resume=True)
    assert p.read_bytes()==b'preserved partial bytes'


def test_derived_last_observation_propagates_missing_dependency_nat():
    queries=q();heads={h:dict(prediction=np.array([1.]),prior_exposure=np.array([10.]),last_day=np.array([9800.]),source_supported=np.array([True]),player_observed=np.array([True])) for h in D.HEAD_NAMES}
    heads['at_rim_pct'].update(prediction=np.array([np.nan]),prior_exposure=np.array([0.]),last_day=np.array([-1.]),player_observed=np.array([False]))
    support=pd.DataFrame(columns=['player_id','game_date','playoffs','off_poss','def_poss','minutes','minute_observed'])
    frame=D.compact_from_heads(queries,heads,D.SupportIndex(support))
    assert pd.isna(frame.hist__rim_fgm100__last_observation_date.iloc[0])
    assert frame.hist__rim_fgm100__missing.iloc[0] and frame.hist__rim_fgm100__exposure.iloc[0]==0
    assert pd.notna(frame.hist__rim_fga100__last_observation_date.iloc[0])


def test_export_nonblocking_lock_refuses_concurrent_resume(tmp_path):
    root=tmp_path/'locked';lock=root.with_name(root.name+'.export.lock')
    with lock.open('a+') as held:
        D.fcntl.flock(held.fileno(),D.fcntl.LOCK_EX|D.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='healthy daily export'):fake_store().export(q(),root,minimum_free_bytes=0,resume=True)
    assert not root.exists()


def test_calendar_lookup_once_per_distinct_date_after_date_and_identity_gates(monkeypatch):
    raw=pd.DataFrame(dict(query_id=np.arange(100),player_id=np.arange(1,101),query_date='1997-01-01',query_season=1997,cohort_vintage=1997))
    original=D.parameter_vintage;calls=[]
    def counted(cut):calls.append(cut);return original(cut)
    monkeypatch.setattr(D,'parameter_vintage',counted)
    actual=D.normalize_queries(raw);assert actual.parameter_vintage.eq(1997).all() and len(calls)==1
    calls.clear()
    for bad in [raw.assign(query_date='2027-01-01'),raw.assign(query_date='1997-01-01T12:00:00'),raw.assign(player_id=0)]:
        with pytest.raises(ValueError):D.normalize_queries(bad)
    assert calls==[]


def test_integer_source_days_are_exact_missing_stays_nat_and_bad_days_reject():
    observed=np.array([-1.,9801.,20000.,20400.])
    actual=D.day_dates(observed)
    expected=pd.DatetimeIndex([pd.NaT,*pd.to_datetime(observed[1:],unit='D',origin='1970-01-01')])
    np.testing.assert_array_equal(actual.to_numpy(),expected.to_numpy())
    for bad in [[np.nan],[np.inf],[-2.],[20000.5],[1e100]]:
        with pytest.raises(ValueError,match='exact source'):D.day_dates(bad)


def test_all_exact_actual_modern_day_arrays_match_saved_parquet():
    root=S.parent/'results/estimated_skills';p=root/'vpm_1997_daily_datetime_diagnostic_v1/actual_saved_day_arrays.npz'
    if not p.exists():pytest.skip('Actual frozen source regression artifact not present in this checkout')
    frame=pd.read_parquet(root/'vpm_1997_daily_features_pilot_v3/2026_20dates.parquet')
    with np.load(p) as arrays:
        assert len(arrays.files)==22
        for field in arrays.files:np.testing.assert_array_equal(D.day_dates(arrays[field]).to_numpy(),frame[field+'__last_observation_date'].to_numpy())
