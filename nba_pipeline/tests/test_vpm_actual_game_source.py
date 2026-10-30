from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vpm_actual_game_source import admit,COUNT_SOURCES,SLOTS


def fixture():
    o=pd.DataFrame([dict(game_id=1,player_id=i,season=2026,playoffs=False,
        team_id=10 if i<=5 else 20,opponent_id=20 if i<=5 else 10,is_home=i<=5,
        game_date='2026-01-01',off_poss=1,def_poss=1,seconds_pbp=10,lineups_clean=True,
        **{'core.'+c:0 for values in COUNT_SOURCES.values() for c in values}) for i in range(1,11)])
    a=pd.DataFrame([dict(game_id=1,game_date='2026-01-01',source_season_end_year=2026,season_phase='RS',
                    actual_points=0,**dict(zip(SLOTS,list(range(1,11))))),
                    dict(game_id=1,game_date='2026-01-01',source_season_end_year=2026,season_phase='RS',
                    actual_points=0,**dict(zip(SLOTS,list(range(6,11))+list(range(1,6)))))])
    return a,pd.DataFrame(dict(game_id=[1])),o


def test_observed_zero_kept_and_exposures_separate():
    a,p,o=fixture();o['off_poss']=9
    d,g,q,s,log=admit(a,p,o,2026)
    assert len(q)==10 and q.native_off_poss.eq(1).all() and q.source_off_poss.eq(9).all()
    assert log.accepted.all() and len(d)==2


@pytest.mark.parametrize('bad',['missing','team','date','count','score'])
def test_whole_game_exclusion(bad):
    a,p,o=fixture()
    if bad=='missing':o=o.iloc[1:]
    if bad=='team':o.loc[0,'team_id']=20
    if bad=='date':o.loc[0,'game_date']='2026-01-02'
    if bad=='count':o.loc[0,'core.fta']=np.nan
    if bad=='score':o.loc[0,'core.ftm']=1;o.loc[0,'core.fta']=1
    d,g,q,s,log=admit(a,p,o,2026)
    assert d.empty and g.empty and not log.accepted.any()


def test_protected_year_rejected():
    a,p,o=fixture()
    with pytest.raises(ValueError,match='2027'):admit(a,p,o,2027)


def test_cutoff_ignores_later_game_counts_and_pricing():
    a,p,o=fixture();later=a.copy();later.game_id=2;later.game_date='2026-02-01'
    oo=o.copy();oo.game_id=2;oo.game_date='2026-02-01'
    all_a=pd.concat([a,later],ignore_index=True);all_o=pd.concat([o,oo],ignore_index=True)
    all_p=pd.DataFrame(dict(game_id=[1,2]))
    first=admit(all_a,all_p,all_o,2026,cutoff='2026-01-02')
    all_o.loc[all_o.game_id.eq(2),'core.ftm']=999999
    all_a.loc[all_a.game_id.eq(2),'actual_points']=1e15
    second=admit(all_a,all_p,all_o,2026,cutoff='2026-01-02')
    for i in [0,1,2,4]:pd.testing.assert_frame_equal(first[i],second[i])
    np.testing.assert_array_equal(first[3],second[3])
