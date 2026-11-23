"""Focused identity/count/actual-result checks; no fitting or rating solve."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];P=R/'nba_pipeline/scripts/vpm_daily/entrant_background.py'
spec=importlib.util.spec_from_file_location('entrant_background_tests',P);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
sys.path.insert(0,str(M.V))
from vpm.harness.entrant_priors import normalize_name

def row(**kwargs):
 r=dict(id=1,player='Rookie One Jr.',year=2026,nba_id=None,birthdate='2006-01-01',games=30,mpg=30.,two_m=100,two_a=200,three_m=40,three_a=100,ftm=50,fta=60,apg=2.,topg=1.,orpg=1.,drpg=3.,spg=1.,bpg=.5)
 return {**r,**kwargs}

def test_exact_college_join_same_original_counts_and_conflict_rejection():
 sources=SimpleNamespace(college=pd.DataFrame(columns=['player_id','season']))
 names=pd.Series({10:'Rookie One Jr.',11:'Rookie One'});first=pd.Series({10:2027,11:2027})
 records,rejected,births=M._add_college(sources,[row()],names,{10,11},first,pd.Timestamp('2026-11-09'),normalize_name)
 assert len(records)==1 and records[0]['player_id']==10 and not rejected
 f=sources.college.iloc[0];assert f.minutes==900 and f.ast==60 and f.two_m==100 and str(f.available_at.date())=='2026-07-01'
 sources=SimpleNamespace(college=pd.DataFrame(columns=['player_id','season']))
 records,rejected,_=M._add_college(sources,[row(nba_id=11)],names,{10,11},first,pd.Timestamp('2026-11-09'),normalize_name)
 assert not records and rejected[0]['reason']=='conflicting_nonnull_nba_id'

def test_current_native_counts_ignore_feature_masks_and_exclude_cut(monkeypatch,tmp_path):
 history=pd.DataFrame(dict(player_id=[99],season=[1997],possessions=[5.],last_game=pd.to_datetime(['1997-04-01'])))
 lines=pd.DataFrame(dict(player_id=[1],season=[2026],game_id=[8],game_date=pd.to_datetime(['2026-04-01']),off_poss=[3.],def_poss=[5.]))
 native=pd.DataFrame([{**{s:i+1 for i,s in enumerate(M.SLOTS)},'game_id':9,'event_num':1,'game_date':'2026-11-08'}, {**{s:i+1 for i,s in enumerate(M.SLOTS)},'game_id':10,'event_num':1,'game_date':'2026-11-09'}])
 def read(p,*a,**k):
  if Path(p)==M.ENTRANTS/'nba_history.parquet':return history.copy()
  if Path(p)==M.MONTHLY:return lines.copy()
  return native.copy()
 monkeypatch.setattr(M.pd,'read_parquet',read);monkeypatch.setattr(M,'_sha',lambda p:'fixture')
 career,first,current,_,meta=M._career(tmp_path,pd.Timestamp('2026-11-09'))
 assert len(current)==10 and (current==.5).all() and meta['current_native_rows']==1
 assert career.groupby('player_id').possessions.sum().loc[1]==4.5 and first.loc[1]==2026
 assert career.groupby('player_id').possessions.sum().loc[99]==5.
