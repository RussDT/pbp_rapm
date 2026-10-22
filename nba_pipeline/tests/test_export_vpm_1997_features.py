from pathlib import Path
import json,subprocess,sys
import numpy as np
import pandas as pd
import pytest
S=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(S))
import export_vpm_1997_features as m
import export_full_history_annual_snapshots as old
from full_skill_history_specs import Skill,SKILLS

@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    out=tmp_path_factory.mktemp('history1997kernel')/'kernel.so'
    subprocess.run(['c++','-O2','-std=c++11','-shared','-fPIC',str(S/'full_skill_history_kernel.cpp'),'-o',str(out)],check=True)
    return out

def query(date='1997-10-01',year=1997,player=1):
    return pd.DataFrame([dict(query_id=f'{date}:{year}:{player}',player_id=player,query_date=date,query_season=year,cohort_vintage=year)])

def recipe(year=1997,last=None):
    return dict(year=year,training_end_season=year-1,status='bootstrap' if last is None else 'annual_past_only_fit',last_training_observation=last,mean=.75,parameters=[1500.,.03,30.,.002,.05,.2])

def panel():
    p=pd.DataFrame(dict(player_id=[1,1,1],game_id=['1','2','3'],game_date=pd.to_datetime(['1996-11-01','1997-02-01','1997-10-01']),season=[1997,1997,1998],**{'core.fta':[4.,2.,99.],'core.ftm':[0.,1.,99.]}))
    p['day']=(p.game_date-pd.Timestamp('1970-01-01')).dt.days
    return p

def ft():return Skill('ft_pct','core.ftm','core.fta','shooting','pct',.75)

def test_annual_dates_and_explicit_recipe_range():
    assert m.annual_report_cut(2020)==pd.Timestamp('2020-11-01')
    assert m.annual_report_cut(1997)==pd.Timestamp('1997-10-01')
    m.normalize_queries(query('2026-10-01',2026))
    for year in [1996,2027]:
        with pytest.raises(ValueError):m.annual_report_cut(year)
        with pytest.raises(ValueError):m.normalize_queries(query(year=year))
    with pytest.raises(ValueError):m.normalize_queries(query('2026-10-02',2026))
    with pytest.raises(ValueError):m.normalize_queries(pd.concat([query(),query()]))
    with pytest.raises(ValueError):m.parse_seasons('1997,1997')
    assert m.parse_seasons('1997:1999')==[1997,1998,1999]


def test_universe_is_canonical_observed_prefix_not_future_addresses():
    p=panel();p.loc[2,['player_id','season']]=[999,1998]
    q=m.build_queries(p,[1997,1998]);assert q.loc[q.query_season.eq(1997),'player_id'].tolist()==[1]
    assert q.loc[q.query_season.eq(1998),'player_id'].tolist()==[1,999]
    assert q.query_date.eq(q.query_season.map(m.annual_report_cut)).all()
    p.loc[2,'season']=2027
    with pytest.raises(ValueError,match='Protected'):m.build_queries(p,[1997])


def test_early_terminal_prefix_zero_history_same_date_and_future_mutation(kernel):
    p=panel();q=pd.concat([query(),query(player=2)],ignore_index=True)
    a,pa=m.replay_snapshot(p,q,[ft()],{'ft_pct':recipe()},kernel)
    changed=p.copy();changed.loc[changed.game_date.ge('1997-10-01'),['core.ftm','core.fta']]=1e6
    b,pb=m.replay_snapshot(changed,q,[ft()],{'ft_pct':recipe()},kernel)
    pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(pa,pb)
    assert np.isfinite(a.ft_pct.iloc[0]) and np.isnan(a.ft_pct.iloc[1])
    assert pa.prior_exposure.tolist()==[6.,0.]
    assert pa.last_observation_date.dropna().lt(pd.Timestamp('1997-10-01')).all()
    m.replay_snapshot(p,query('1997-01-01'),[ft()],{'ft_pct':recipe()},kernel)
    again,_=m.replay_snapshot(p,q,[ft()],{'ft_pct':recipe()},kernel)
    pd.testing.assert_frame_equal(a,again)


def test_actual_fit_date_and_epoch_support_stay_strict(kernel):
    with pytest.raises(ValueError,match='training observation'):
        m.replay_snapshot(panel(),query(),[ft()],{'ft_pct':recipe(last='1997-10-01')},kernel)
    spec=Skill('later','core.ftm','core.fta','shooting','pct',.75,first_season=2014)
    v,p=m.replay_snapshot(panel(),query(),[spec],{'later':recipe()},kernel)
    assert np.isnan(v.later.iloc[0]) and not p.player_observed.iloc[0] and not p.source_supported.iloc[0]
    expired=Skill('expired','core.ftm','core.fta','shooting','pct',.75,last_season=1997)
    v,p=m.replay_snapshot(panel(),query('1998-10-01',1998),[expired],{'expired':recipe(1998)},kernel)
    assert np.isnan(v.expired.iloc[0]) and p.player_observed.iloc[0] and not p.source_supported.iloc[0]


def test_modern_replay_is_bit_exact_frozen_implementation(kernel):
    p=panel();p.game_date=pd.to_datetime(['2025-04-01','2025-11-01','2026-01-01']);p.season=[2025,2026,2026];p.day=(p.game_date-pd.Timestamp('1970-01-01')).dt.days
    q=query('2026-09-01',2026);r={'ft_pct':recipe(2026,'2025-06-01')}
    a,pa=m.replay_snapshot(p,q,[ft()],r,kernel);b,pb=old.replay_snapshot(p,q,[ft()],r,kernel)
    pd.testing.assert_frame_equal(a,b,check_exact=True);pd.testing.assert_frame_equal(pa,pb,check_exact=True)


def generic(date='1997-10-01',year=1997):
    q=m.normalize_queries(query(date,year));values=pd.DataFrame({s.name:[.5 if s.kind=='pct' else 1.] for s in SKILLS});heads=[]
    for s in SKILLS:
        d=q.iloc[0].to_dict();d.update(head_name=s.name,prior_exposure=10.,last_observation_date=pd.Timestamp(date)-pd.Timedelta(days=3),source_supported=True,player_observed=True);heads.append(d)
    return m.make_features(values,pd.DataFrame(heads),q,m.feature_schema()),q,values,pd.DataFrame(heads)


def test_generic_modern_derived_aliases_and_missing_v2_are_unchanged():
    _,q,values,heads=generic('2026-09-01',2026)
    heads.loc[heads.head_name.eq('dreb_contest100__v2'),['source_supported','player_observed','prior_exposure']]=[False,False,0.]
    values['dreb_contest100__v2']=np.nan
    a=m.make_features(values,heads,q,m.feature_schema());b=old.make_features(values,heads,q,old.feature_schema())
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    assert a['lib__dreb_contest100'].isna().all()


def minute_panel(year=1997):
    p=pd.DataFrame(dict(player_id=np.arange(1,11),game_id='0000000001',season=year,game_date=pd.Timestamp(f'{year}-05-01'),playoffs=False,team_id=[1]*5+[2]*5,seconds_pbp=2880.,lineups_clean=True,known_issue=None,off_poss=20.,def_poss=21.))
    return p

def empty_validated():return pd.DataFrame(columns=['player_id','game_id','game_date','minutes'])


def test_minutes_gate_rejects_bad_clock_known_issues_and_preserves_modern_mask():
    p=minute_panel();s,r=m.observation_support(p,empty_validated());assert s.minute_observed.all() and s.minutes.eq(48.).all()
    bad=p.copy();bad.loc[0,'seconds_pbp']=2879.
    s,r=m.observation_support(bad,empty_validated());assert s.minutes.isna().all() and r['early_excluded_games']==1
    bad=p.copy();bad.loc[0,'known_issue']='unresolved';s,_=m.observation_support(bad,empty_validated());assert s.minutes.isna().all()
    p=minute_panel(2014);v=p[['player_id','game_id','game_date']].copy();v['minutes']=48.;v.loc[0,'minutes']=np.nan
    s,r=m.observation_support(p,v);assert np.isnan(s.minutes.iloc[0]) and not s.minute_observed.iloc[0] and s.minutes.iloc[1]==48.
    v.loc[1,'minutes']=47.
    with pytest.raises(ValueError,match='definitions differ'):m.observation_support(p,v)


def test_compact_units_missing_physical_zero_and_literal_mpg_are_explicit():
    wide,q,_,_=generic();p=minute_panel();support,_=m.observation_support(p,empty_validated())
    wide.loc[0,'lib__oreb100']=0.
    out=m.compact_features(wide,m.feature_schema(),support)
    assert out.hist__oreb100.iloc[0]==0. and not out.hist__oreb100__missing.iloc[0]
    assert out.hist__nonoff_pf100_total.isna().all()
    assert out.ctx_mpg.iloc[0]==48. and out.ctx_games.iloc[0]==1
    assert out.archive_off_possessions_lower_bound.iloc[0]==20. and out.archive_def_possessions_lower_bound.iloc[0]==21.
    assert out.estimate_date.eq(out.query_date).all() and out.last_archive_observation_date.lt(out.query_date).all()
    schema=m.compact_schema(m.feature_schema());records={r['name']:r for r in schema['records']}
    assert records['hist__pf100_total']['units']=='per_100_total_possessions'
    assert 'technical' in records['hist__pf100_total']['description']
    assert records['hist__shooting_pf100_total']['source']=='lib__foul_shooting100'
    assert not schema['records'][-2].get('source')
    changed=support.copy();changed.game_date=pd.Timestamp('1997-10-01');out=m.compact_features(wide,m.feature_schema(),changed)
    assert out.ctx_mpg.isna().all() and out.ctx_games.eq(0).all() and out.archive_observed_games.eq(0).all()


def test_compact_current_day_history_is_rejected():
    wide,_,_,_=generic();wide['lib__oreb100__last_observation_date']=pd.Timestamp('1997-10-01')
    with pytest.raises(ValueError,match='reaches its query'):m.compact_features(wide,m.feature_schema(),m.observation_support(minute_panel(),empty_validated())[0])
