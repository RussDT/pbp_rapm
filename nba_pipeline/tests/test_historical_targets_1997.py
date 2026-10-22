"""Early actual-target identity, scope, missingness and immutable-source gates."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import historical_targets_1997 as M


def evidence():
    raw=pd.DataFrame(dict(game_id=[29600021]*3,player1_id=[471]*3,
        player1_team_id=[1610612758]*3,game_date=['1996-11-02']*3,
        event_type=[1,3,8],home_description=['Simmons 3PT Jump Shot',
            'MISS Simmons Free Throw 1 of 1','SUB out: nan'],visitor_description=['']*3))
    core=pd.DataFrame([dict(game_id=29600021,person_id=1489,team_id=1610612758,
        lineups_clean=True,provably_wrong_rows=0,rows_without_10_players=0,
        season=1997,playoffs=False,date=19961102,pts=3,fga=1,fgm=1,
        fg3a=1,fg3m=1,fta=1,ftm=0,tov=0)])
    official=pd.DataFrame([dict(PLAYER_ID=1489,PLAYER_NAME='Lionel Simmons',
        TEAM_ID=1610612758,GP=41)])
    return raw,core,official


def prove(raw,core,official):
    return M.prove_alias(raw,core,official,471,1489,'Lionel Simmons','Simmons',1610612758)


def test_exact_dated_alias_requires_native_counts_and_official_team():
    raw,core,official=evidence();rows=prove(raw,core,official)
    assert rows[0]['counts']==dict(pts=3,fga=1,fgm=1,fg3a=1,fg3m=1,fta=1,ftm=0,tov=0)
    assert rows[0]['substitution_actor_rows']==1
    core.loc[0,'pts']=2
    with pytest.raises(ValueError,match='counts differ'):prove(raw,core,official)


@pytest.mark.parametrize('change',['date','team','unsafe','duplicate','wrong_epoch','surname'])
def test_alias_identity_not_a_surname_or_partial_source_guess(change):
    raw,core,official=evidence()
    if change=='date':core.loc[0,'date']=19961103
    if change=='team':official.loc[0,'TEAM_ID']=1610612744
    if change=='unsafe':core.loc[0,'lineups_clean']=False
    if change=='duplicate':core=pd.concat([core,core],ignore_index=True)
    if change=='wrong_epoch':core.loc[0,'season']=1998
    if change=='surname':raw.loc[0,'home_description']='Other 3PT Jump Shot'
    with pytest.raises(ValueError):prove(raw,core,official)


def lineup(game=29600021):
    return dict(game_id=game,**dict(zip(M.SLOTS,[471,2,3,4,5,6,7,8,9,10])))


def test_alias_scope_does_not_rewrite_other_games_years_or_raw_frame():
    a=pd.DataFrame([lineup(),lineup(29600022)])
    packet=dict(aliases=[dict(year=1997,phase='RS',game_id=29600021,
        source_id=471,canonical_id=1489,admitted=True)])
    b=M.apply_aliases(a,1997,'RS',packet)
    assert b.O1.tolist()==[1489,471] and a.O1.tolist()==[471,471]
    assert M.apply_aliases(a,1998,'RS',packet).equals(a)
    assert M.apply_aliases(a,1997,'PS',packet).equals(a)
    packet['aliases'][0]['canonical_id']=2
    with pytest.raises(ValueError,match='colliding'):M.apply_aliases(a,1997,'RS',packet)


def test_raw_actual_ft_and_missi_word_boundary():
    d=pd.DataFrame(dict(event_type=[1,1,3,3],home_description=[
        'Simmons 3PT Jump Shot','Booker Layup','Missi Free Throw 1 of 2',
        'MISS Missi Free Throw 2 of 2'],visitor_description=['']*4))
    assert M.raw_counts(d)==dict(pts=6,fga=2,fgm=2,fg3a=1,fg3m=1,fta=2,ftm=1,tov=0)


def test_identity_censor_is_row_based_raw_pricing_is_not_reconstructed():
    rows=[]
    for game,event,ids in [(1,1,range(1,11)),(1,2,[99,*range(2,11)]),(2,1,[0,*range(2,11)])]:
        rows.append(dict(game_id=game,event_num=event,source_season_end_year=1997,
            season_phase='RS',game_date='1997-01-01',actual_points=2,
            First_Chance_Diff=2,Second_Chance_Diff=0,**dict(zip(M.SLOTS,ids))))
    atoms=pd.DataFrame(rows);games=pd.DataFrame(dict(game_id=[1,2],source_season_end_year=[1997]*2,
        season_phase=['RS']*2,game_date=['1997-01-01']*2,raw_first_chance_attempts=[99,55]))
    d,g,mask=M.identity_filter(atoms,games,dict(seasons={'1997':dict(admitted_ids=list(range(1,11)))}))
    assert len(d)==1 and mask.tolist()==[True,False,False]
    assert g.raw_first_chance_attempts.tolist()==[99]


def test_immutable_bytes_and_current_pin_verification(tmp_path):
    p=tmp_path/'receipt.json';M.write_once(p,dict(status='passed'));M.write_once(p,dict(status='passed'))
    with pytest.raises(ValueError,match='Immutable'):M.write_once(p,dict(status='different'))
    pins={str(p):M.H.sha(p)};M.verify_pins(pins);p.write_text('{}')
    with pytest.raises(ValueError,match='changed'):M.verify_pins(pins)


def test_same_day_and_future_pricing_rejected_by_shared_actual_target_gate():
    a=pd.DataFrame([dict(game_id=1,event_num=1,source_season_end_year=1997,
        season_phase='RS',game_date='1997-01-01',actual_points=2,
        First_Chance_Diff=2,Second_Chance_Diff=0,**dict(zip(M.SLOTS,range(1,11))))])
    g=a[M.H.KEYS+['game_date']].copy()
    with pytest.raises(ValueError,match='date fence'):M.validate_population(a,g,'1997-01-01')
    M.validate_population(a,g,'1997-01-02')


def test_source_census_uses_canonical_dates_when_raw_has_no_date_field(tmp_path):
    raw=tmp_path/'raw.parquet';canonical=tmp_path/'canonical.parquet'
    pd.DataFrame(dict(game_id=['1','1','2'])).to_parquet(raw,index=False)
    pd.DataFrame(dict(game_id=[1,2],game_date=['1997-01-01','1997-01-02'])).to_parquet(canonical,index=False)
    row=M.source_census_row(raw,canonical,1997,'RS')
    assert row['raw_rows']==3 and row['canonical_games']==2
    assert row['first_date']=='1997-01-01'
    pd.DataFrame(dict(game_id=[1,3],game_date=['1997-01-01','1997-01-02'])).to_parquet(canonical,index=False)
    with pytest.raises(ValueError,match='census'):M.source_census_row(raw,canonical,1997,'RS')

