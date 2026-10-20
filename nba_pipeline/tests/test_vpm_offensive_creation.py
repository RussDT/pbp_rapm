import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_vpm_offensive_creation import resolve_passer,mix_states,CATS
from analyze_pbp_playtype_difficulty_ots import classify_pbp_shots


def test_passer_requires_exact_own_side_identity_and_rejects_ambiguity():
    anchors={('g','H','ja. green'):1,('g','A','green'):3};names={1:'green',2:'green',3:'green'}
    assert resolve_passer('g','H','Ja. Green',[1,2],anchors,names)==1
    assert resolve_passer('g','H','Green',[1,2],anchors,names) is None
    assert resolve_passer('g','H','Ja. Green',[2,3],anchors,names) is None


def test_shot_mix_is_coherent_and_later_games_do_not_change_past_states():
    d=pd.DataFrame(dict(player_id=[1,1,2,2],game_date=pd.to_datetime(['2020-01-01','2020-02-01','2020-01-02','2020-03-01']),game_id=['1','2','3','4']));rng=np.random.default_rng(58)
    for c in CATS:d[c]=rng.integers(0,4,len(d))
    pop=np.ones(6)/6;a,pre,post=mix_states(d,180,50,pop);np.testing.assert_allclose(post.sum(axis=1),1);assert (post>=0).all()
    prefix=d.loc[d.game_date.lt('2020-02-15')];b,_,short=mix_states(prefix,180,50,pop);ix=pd.MultiIndex.from_frame(a[['player_id','game_id']]).get_indexer(pd.MultiIndex.from_frame(b[['player_id','game_id']]));np.testing.assert_allclose(short,post[ix],atol=1e-14)


def test_playtype_taxonomy_does_not_use_makes_or_assist_tags():
    d=pd.DataFrame(dict(home_description=['A Driving Layup (2 PTS) (B 1 AST)','A Step Back 3PT Jump Shot (3 PTS)'],visitor_description=['','']));a=classify_pbp_shots(d,pd.Series([False,True]));e=d.copy();e.home_description=['MISS A Driving Layup','MISS A Step Back 3PT Jump Shot'];pd.testing.assert_series_equal(a,classify_pbp_shots(e,pd.Series([False,True])))
