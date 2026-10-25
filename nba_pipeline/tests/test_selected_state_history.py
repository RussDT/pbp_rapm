from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from selected_state_history import (held_snapshot,tracking_snapshot,mpg_snapshot,
 normalize_schedule,vintage_at,compatible_early_packet)


def states():
 return pd.DataFrame(dict(player_id=[1,1,2],game_id=['a','b','c'],game_date=pd.to_datetime(['2020-01-01','2020-01-03','2020-01-01']),value=[0.,9.,2.],state_variance=[1.,1.,1.],observed_exposure=[10.,20.,10.]))


def test_held_states_strict_prefix_zero_density_and_unknown_id():
 s=states();a=held_snapshot(s,[1],'2020-01-03');b=held_snapshot(s,[999,2,1],'2020-01-03')
 assert a.value.item()==b.loc[b.player_id.eq(1),'value'].item()==0
 assert pd.isna(b.loc[b.player_id.eq(999),'value'].item())
 s.loc[s.game_date.ge('2020-01-03'),'value']=1e9
 assert held_snapshot(s,[1],'2020-01-03').value.item()==0


def test_stale730_boundary_and_source_not_mutated():
 s=states().iloc[[0]].copy();before=s.copy(deep=True);date=s.game_date.iloc[0]
 assert held_snapshot(s,[1],date+pd.Timedelta(days=730)).value.item()==0
 assert pd.isna(held_snapshot(s,[1],date+pd.Timedelta(days=731)).value.item())
 pd.testing.assert_frame_equal(s,before)


def test_tracking_original_population_and_observed_zero():
 m=pd.DataFrame(dict(player_id=[1,2,1],game_date=pd.to_datetime(['2020-01-01','2020-01-01','2020-01-03']),feature='def_contested_shots100',numerator=[0.,10.,1000.],denominator=100.))
 a,p=tracking_snapshot(m,[1,999],'2020-01-03');b,_=tracking_snapshot(m,[2,1],'2020-01-03')
 decay=2**(-2/550);expected=100*(1000*.05)/(100*decay+1000)
 assert a.def_contested_shots100.iloc[0]==pytest.approx(expected)
 assert a.def_contested_shots100.iloc[0]==b.def_contested_shots100.iloc[1]
 assert pd.isna(a.def_contested_shots100.iloc[1]);assert not p.loc[p.feature.eq('def_contested_shots100')].is_missing.iloc[0]


def test_mpg_positive_RS_exact365_window():
 d=pd.DataFrame(dict(player_id=1,game_id=['a','b','c','d','e'],game_date=pd.to_datetime(['2019-01-02','2020-01-01','2020-01-01','2020-01-02','2019-01-01']),season_type=['rs','rs','ps','rs','rs'],minutes=[10.,0.,40.,50.,90.]))
 x=mpg_snapshot(d,[1,2],'2020-01-02');assert x.ctx_mpg.iloc[0]==10.;assert x.ctx_games.iloc[0]==1;assert np.isnan(x.ctx_mpg.iloc[1])


def test_explicit_native_season_and_legal_annual_calendar():
 q=pd.DataFrame(dict(query_id=['q'],player_id=[1],query_date=['2020-11-01'],query_season=[2020]));assert normalize_schedule(q).query_season.item()==2020
 assert vintage_at('2020-11-01')==2019;assert vintage_at('2020-12-01')==2020;assert vintage_at('2015-09-30') is None


@pytest.mark.parametrize('field,value',[('player_id',1.5),('query_season',2027),('query_date','2026-10-02'),('query_date','2020-01-01T01:00:00')])
def test_bad_query_rejected(field,value):
 q=pd.DataFrame(dict(query_id=['q'],player_id=[1],query_date=['2020-01-01'],query_season=[2020]));q[field]=value
 with pytest.raises(ValueError):normalize_schedule(q)


def test_early_compatible_zero_has_support_and_no_foul_rename():
 d=pd.DataFrame({'player_id':[1,2],'query_date':pd.to_datetime(['2000-01-01']*2),'hist__stl100':[0.,9.],'hist__stl100__exposure':[10.,0.],'hist__stl100__last_observation_date':pd.to_datetime(['1999-12-01']*2),'hist__pf100_total':[3.,3.],'hist__shooting_pf100_total':[1.,1.]})
 p,h=compatible_early_packet(d);assert p.own_stl_rate.iloc[0]==0;assert np.isnan(p.own_stl_rate.iloc[1]);assert 'own_pf_committed_rate' not in p;assert 'def_shooting_pf100' not in p
 d.loc[0,'hist__stl100__last_observation_date']=pd.Timestamp('2000-01-01')
 with pytest.raises(ValueError):compatible_early_packet(d)


def test_era_join_never_fills_modern_stale_state_or_overwrites_selected_zero():
 from selected_state_history import combine_eras
 a=pd.DataFrame({'player_id':[1,1,2],'estimate_date':pd.to_datetime(['2000-01-01','2026-09-01','2000-01-01']),'own_stl_rate':[np.nan,np.nan,0.]})
 b=a[['player_id','estimate_date']].copy();b['own_stl_rate']=9.
 x=combine_eras(a,b);assert x.own_stl_rate.iloc[0]==9;assert np.isnan(x.own_stl_rate.iloc[1]);assert x.own_stl_rate.iloc[2]==0


def test_mask_control_removes_phantom_same_day_and_only_registered_fields():
 from selected_state_history import mask_legacy_source,MASK_FIELDS
 d=pd.DataFrame({'player_id':[1,999,1],'game_id':['a','a','b'],'game_date':pd.to_datetime(['2020-01-01','2020-01-01','2020-01-03']),'off_poss':[100.]*3,'def_poss':[100.]*3})
 for field in MASK_FIELDS:d[field]=[0.,1e12,5.]
 d['usg_events']=d.fga+.44*d.fta+d.tov
 f=pd.DataFrame({'player_id':[999],'game_id':['a']});m=pd.DataFrame({'player_id':[1],'game_id':['b'],'field':['stl'],'reason':['typed_uncertain'],'source_value':[999.]})
 a,ctx,r=mask_legacy_source(d,f,m,context=d[['player_id','game_id']]);assert set(a.player_id)=={1};assert len(ctx)==2;assert a.off_poss.eq(100).all();assert np.isnan(a.stl.iloc[1]);assert a.blk.iloc[1]==5
 before=a.loc[a.game_date.lt('2020-01-03')].copy();d.loc[d.game_date.ge('2020-01-03'),'blk']=999
 b,_,_=mask_legacy_source(d,f,m);pd.testing.assert_frame_equal(before,b.loc[b.game_date.lt('2020-01-03')]);assert not r['denominators_changed'];assert not r['validated_counter_values_copied']


def test_mask_control_rejects_wrong_full_census_and_foul_or_denominator_alias():
 from selected_state_history import mask_legacy_source,MASK_FIELDS
 d=pd.DataFrame({'player_id':[1],'game_id':['a'],'off_poss':[1.],'def_poss':[1.]})
 for f in MASK_FIELDS:d[f]=1.
 foreign=d[['player_id','game_id']];m=pd.DataFrame({'player_id':[1],'game_id':['a'],'field':['off_poss'],'reason':['x'],'source_value':[1.]})
 with pytest.raises(ValueError):mask_legacy_source(d,foreign,m,require_full_census=True)
 with pytest.raises(ValueError):mask_legacy_source(d,foreign,m)


def test_rim_quality_support_exact_poststate_decay_and_zero_attempt():
 from selected_state_history import decayed_rim_quality_support
 d=pd.DataFrame({'player_id':[1,1,1,2],'game_date':pd.to_datetime(['2020-01-01','2020-01-03','2020-01-05','2020-01-05']),'rim_dfga':[4.,0.,3.,0.]})
 np.testing.assert_allclose(decayed_rim_quality_support(d,2),[4.,2.,4.,0.],atol=0,rtol=0)


def test_cache_fingerprint_independent_of_previously_requested_other_models(tmp_path):
 from selected_state_history import SelectedStateHistory,sha
 core=tmp_path/'core';path=core/'models/2025/a.json';path.parent.mkdir(parents=True);path.write_text('{}');(path.parent/'b.json').write_text('{}')
 source=tmp_path/'source';source.write_text('fixed')
 store=SelectedStateHistory.__new__(SelectedStateHistory);store.cache=tmp_path/'cache';store.cache.mkdir();store.core=core;store.source_mode='legacy_anchor';store.defense=tmp_path/'unused';store._states={};store._inputs={};store._source_inputs={str(source):sha(source)};store._model=lambda y,s:{};calls=[]
 def make(y,s,m):calls.append(s);return states()
 store._core_states=make
 store.poststates(2025,'a');store.poststates(2025,'b');store._states.clear();store._inputs['other_model']='semantic_previous_request';store.poststates(2025,'a');assert calls==['a','b']
 # Actual modified cache bytes cannot be silently resumed.
 (store.cache/'2025/a.parquet').write_bytes(b'changed');store._states.clear()
 with pytest.raises(ValueError):store.poststates(2025,'a')


def test_source_read_guard_rejects_hash_read_race_and_wrong_expected_bytes(tmp_path):
 from selected_state_history import SelectedStateHistory,sha
 store=SelectedStateHistory.__new__(SelectedStateHistory);store._inputs={};p=tmp_path/'source';p.write_text('old');old=sha(p)
 def changing(path):
  data=path.read_text();path.write_text('new');return data
 with pytest.raises(ValueError):store._read_pinned(p,changing)
 store._inputs={}
 with pytest.raises(ValueError):store._read_pinned(p,lambda x:x.read_text(),expected=old)
 store._inputs={};assert store._read_pinned(p,lambda x:x.read_text(),expected=sha(p))=='new'
